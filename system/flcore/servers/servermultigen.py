import copy
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clientmultigen import clientMultiGen
from flcore.servers.serverbase import Server
from threading import Thread
import wandb

class FedMultiGen(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientMultiGen)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.learning_rate_decay = args.learning_rate_decay
        self.generative_models = [
            Generative(
                args.noise_dim,
                args.num_classes,
                args.hidden_dim,
                self.clients[i].feature_dim,
                self.device,
            ).to(self.device)
            for i in range(self.num_clients)
        ]
        self.generative_optimizers = [
            torch.optim.Adam(
                params=gen_model.parameters(),
                lr=args.generator_learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                amsgrad=False,
            )
            for gen_model in self.generative_models
        ]
        self.generative_learning_rate_schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=args.learning_rate_decay_gamma
            )
            for optimizer in self.generative_optimizers
        ]
        self.loss = nn.CrossEntropyLoss()
        self.global_loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.global_model.parameters(), lr=self.learning_rate)
        self.global_batch_size = args.global_batch_size
        self.global_epochs = args.global_epochs
        for client in self.clients:
            for yy in range(self.num_classes):
                client.qualified_labels.extend(
                    [yy for _ in range(int(client.sample_per_class[yy].item()))]
                )

        self.server_epochs = args.server_epochs
        self.localize_feature_extractor = args.localize_feature_extractor
        if self.localize_feature_extractor:
            self.global_model = copy.deepcopy(args.model.head)
        self.use_wandb = args.use_wandb

    def train(self):
        for i in range(self.global_rounds + 1):
            glob_iter = i
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate(glob_iter)

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.train_generator(glob_iter)
            self.aggregate_parameters()
 
            self.train_global_model()

            self.Budget.append(time.time() - s_t)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientGen)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def send_models(self):
        assert len(self.clients) > 0

        for i, client in enumerate(self.clients):
            start_time = time.time()

            client.set_parameters(self.global_model, self.generative_models[i])

            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients,
            int((1 - self.client_drop_rate) * self.current_num_join_clients),
        )

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = (
                    client.train_time_cost["total_cost"]
                    / client.train_time_cost["num_rounds"]
                    + client.send_time_cost["total_cost"]
                    / client.send_time_cost["num_rounds"]
                )
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                if self.localize_feature_extractor:
                    self.uploaded_models.append(client.model.head)
                else:
                    self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def train_generator(self, glob_iter):
        for i, client_id in enumerate(self.uploaded_ids):
            self.generative_models[client_id].train()
            for _ in range(self.server_epochs):
                labels = np.random.choice(
                    self.clients[client_id].qualified_labels, self.batch_size
                )
                labels = torch.LongTensor(labels).to(self.device)
                z = self.generative_models[client_id](labels)

                logits = 0
                model = self.uploaded_models[i]
                model.eval()
                if self.localize_feature_extractor:
                    logits += model(z)
                else:
                    logits += model.head(z)

                self.generative_optimizers[client_id].zero_grad()
                loss = self.loss(logits, labels)
                loss.backward()
                self.generative_optimizers[client_id].step()
            if self.learning_rate_decay:
                self.generative_learning_rate_schedulers[client_id].step()

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(
                self.global_model, self.generative_model, self.qualified_labels
            )
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    def train_global_model(self):
        # train global model with synthetic data from generators 
        # create a synthetic dataset
        synthetic_data = []
        synthetic_labels = []
        for client_id in self.uploaded_ids:
            labels = np.random.choice(self.clients[client_id].qualified_labels, self.batch_size)
            labels = torch.LongTensor(labels).to(self.device)
            z = self.generative_models[client_id](labels)
            synthetic_data.append(z)
            synthetic_labels.append(labels)
        synthetic_data = torch.cat(synthetic_data, dim=0)
        synthetic_labels = torch.cat(synthetic_labels, dim=0)
        # divide into batches
        synthetic_dataset = torch.utils.data.TensorDataset(synthetic_data, synthetic_labels)
        synthetic_dataloader = torch.utils.data.DataLoader(synthetic_dataset, batch_size=self.global_batch_size, shuffle=True) 
        # train global model
        self.global_model.train()   
        for epoch in range(self.global_epochs):
            for (x, y) in synthetic_dataloader:
                if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.global_model.head(x)
                loss = self.global_loss(output, y)
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()


# based on official code https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/trainmodel/generator.py
class Generative(nn.Module):
    def __init__(self, noise_dim, num_classes, hidden_dim, feature_dim, device) -> None:
        super().__init__()

        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.device = device

        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim + num_classes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(self, labels):
        batch_size = labels.shape[0]
        eps = torch.rand(
            (batch_size, self.noise_dim), device=self.device
        )  # sampling from Gaussian

        y_input = F.one_hot(labels, self.num_classes)
        z = torch.cat((eps, y_input), dim=1)

        z = self.fc1(z)
        z = self.fc(z)

        return z
