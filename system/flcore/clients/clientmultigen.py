import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
import wandb

class clientMultiGen(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        trainloader = self.load_train_data()
        for x, y in trainloader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                rep = self.model.base(x).detach()
            break
        self.feature_dim = rep.shape[1]

        self.sample_per_class = torch.zeros(self.num_classes)
        trainloader = self.load_train_data()
        for x, y in trainloader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1

        self.qualified_labels = []
        self.generative_model = None
        self.localize_feature_extractor = args.localize_feature_extractor
        self.kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.use_wandb = args.use_wandb
        self.cold_start = args.cold_start
        

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
            
        
    def set_parameters(self, model, generative_model):
        if self.localize_feature_extractor:
            for new_param, old_param in zip(model.parameters(), self.model.head.parameters()):
                old_param.data = new_param.data.clone()
        else:
            for new_param, old_param in zip(model.parameters(), self.model.parameters()):
                old_param.data = new_param.data.clone()

        self.generative_model = generative_model

    def train_metrics(self, glob_iter):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        LATENT_LOSS = 0.0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                p_output = torch.log_softmax(output, dim=1).clone().detach()
                loss = self.loss(output, y)
                # only compute latent loss after first global iteration because the generator is not trained yet
                if glob_iter > 0:
                    z = self.generative_model(y)['output']
                    y_given_gen = self.model.head(z)
                    p_output_given_gen = torch.softmax(y_given_gen, dim=1).clone().detach()
                    # Compute the KL divergence loss
                    latent_loss = self.kl_loss(p_output, p_output_given_gen)
                    LATENT_LOSS += latent_loss.item() * y.shape[0]
                
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
  

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        if self.use_wandb:
            log_key = f'Client_{self.id}/Train_Loss'
            wandb.log({log_key: losses / train_num}, step=glob_iter)
            if glob_iter > 0:
                LATENT_LOSS = LATENT_LOSS / train_num
                log_key = f'Client_{self.id}/Latent_Loss'
                wandb.log({log_key: LATENT_LOSS}, step=glob_iter)
        return losses, train_num
