import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class ResidualBlock(nn.Module):
    def __init__(self, dim, use_bn=True):
        super().__init__()
        layers = [
            spectral_norm(nn.Linear(dim, dim)),
        ]
        if use_bn:
            layers += [nn.BatchNorm1d(dim)]
        layers += [nn.LeakyReLU(0.2, inplace=True),
                   spectral_norm(nn.Linear(dim, dim))]
        if use_bn:
            layers += [nn.BatchNorm1d(dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.leaky_relu(x + self.net(x), 0.2)

class ConditionalFiLM(nn.Module):
    """
    Simple FiLM module: produce gamma, beta from label embedding and apply
    per-feature affine transform: gamma * x + beta
    """
    def __init__(self, emb_dim, out_dim):
        super().__init__()
        self.gamma = nn.Linear(emb_dim, out_dim)
        self.beta = nn.Linear(emb_dim, out_dim)

    def forward(self, x, emb):
        g = self.gamma(emb)
        b = self.beta(emb)
        return g * x + b

class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """
    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))


class Generative(nn.Module):
    def __init__(self, noise_dim, num_classes, hidden_dim, feature_dim, device,
                 n_resblocks=3, use_spectral=True, use_bn=True, embedding_dim=None):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.device = device
        self.feature_dim = feature_dim
        self.diversity_loss = DiversityLoss(metric='l1')
        if embedding_dim is None:
            embedding_dim = max(32, num_classes)  # keep it flexible
        self.label_emb = nn.Embedding(num_classes, embedding_dim)

        in_dim = noise_dim + embedding_dim
        # initial projection
        self.proj = nn.Sequential(
            spectral_norm(nn.Linear(in_dim, hidden_dim)) if use_spectral else nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if use_spectral else nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # residual stack
        self.resblocks = nn.ModuleList([ResidualBlock(hidden_dim, use_bn=use_bn) for _ in range(n_resblocks)])
        # FiLM conditioning at the end
        self.film = ConditionalFiLM(embedding_dim, hidden_dim)

        # output head
        self.head = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if use_spectral else nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(hidden_dim, feature_dim)) if use_spectral else nn.Linear(hidden_dim, feature_dim),
            # no activation here — let the training objective decide (or add Tanh if features are normalized)
        )

        # small skip connection from noise->output for easier optimization
        self.skip = nn.Linear(noise_dim, feature_dim)

    def forward(self, labels, verbose=True):
        result = {}
        batch_size = labels.shape[0]
        eps = torch.randn((batch_size, self.noise_dim), device=self.device)  # gaussian
        emb = self.label_emb(labels)  # (B, embedding_dim)
        z = torch.cat((eps, emb), dim=1)
        x = self.proj(z)
        for r in self.resblocks:
            x = r(x)
        x = self.film(x, emb)
        out = self.head(x) + self.skip(eps) * 0.1  # small skip scale
        result['output'] = out
        if verbose:
            result['eps'] = eps
        return result
