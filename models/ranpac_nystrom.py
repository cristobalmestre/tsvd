import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet,SimpleCosineIncrementalNet,MultiBranchCosineIncrementalNet,SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy, sketch_initialization, linear_update, fixed_rank_psd_approximation, cholesky_update, cholesky_update_batch, test_cholesky_update, optimized_cholesky_update_batch, batched_cholesky_update_diag_vectorized

import time
import os

# tune the model at first session with adapter, and then conduct simplecil.
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self. batch_size = args["batch_size"]
        self. init_lr = args["init_lr"]
        
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args

        self.RP_initialized = False

        self.times = {}
        self.times['feature'] = 0
        self.times['algorithm'] = 0

    def after_task(self):
        self._known_classes = self._total_classes

    def save_times(self):
        M = self.args['M']

        root = './results'
        if not os.path.exists(root):
            os.makedirs(root)

        dataset_name = self.args['dataset']
        backbone_name = self.args['backbone_type']
        model_name = self.args['model_name']
        inc = self.args['increment']

        fn = f'{root}/training_time-inc{inc}-M={M}-{model_name}-{backbone_name}-{dataset_name}.npy'
        np.save(fn, [self.times['feature'], self.times['algorithm']])

        # Log the times in INFO level
        logging.info(f"Feature extraction time: {self.times['feature']:.4f} seconds")
        logging.info(f"Algorithm processing time: {self.times['algorithm']:.4f} seconds")
        logging.info(f"Total training time saved in: {fn}")

    def replace_fc(self, trainloader, model, args):

        # Set parameters directly instead of retrieving from args
        use_nystrom = True  # Whether to use Nyström approximation
        sketch_size = 2 * self.args["nb_classes"]  # Size of the sketch (k parameter)
        nystrom_rank = self.args["nb_classes"]  # Target rank (r parameter)
        ridge = 100000  # Ridge regularization parameter


        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                torch.cuda.synchronize()
                feature_start = time.time()

                (_,data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.extract_vector(data)

                torch.cuda.synchronize()
                self.times['feature'] += time.time() - feature_start

                algorithm_start = time.time()
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
                self.times['algorithm'] += time.time() - algorithm_start

        algorithm_start = time.time()

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        # Move to device once for consistent processing
        embedding_list = embedding_list.to(self._device)
        label_list = label_list.to(self._device)
        
        # Create one-hot encoded labels
        labels_one_hot = target2onehot(label_list, self.args["nb_classes"])

        #Y = target2onehot(label_list, self.args["nb_classes"])

        # Make sure W_rand is on the device
        self.W_rand = self.W_rand.to(self._device)

        #Features_h = F.relu(embedding_list @ self.W_rand.cpu())
        Features_h = F.relu(embedding_list @ self.W_rand)
        Features_h_T = Features_h.T

        # Check if the first time running (need to initialize self.Y with one-hot labels)
        if not hasattr(self, 'Y_initialized') or not self.Y_initialized:
            # First time running - initialize self.Y with one-hot labels
            self.Y = labels_one_hot
            self.Y_initialized = True
            
            # Also initialize Q (since it's the first time)
            self.Q = self.Q.to(self._device)
            self.Q = Features_h_T @ labels_one_hot
        else:
            # Not the first time - update Q normally
            self.Q = self.Q.to(self._device)
            self.Q = self.Q + Features_h_T @ labels_one_hot

        # Make sure Q and G are on the same device
        #self.Q = self.Q.to(self._device)
        # G should go away
        #self.G = self.G.to(self._device)

        #self.Q = self.Q + Features_h_T @ self.Y
        #self.G = self.G + Features_h_T @ Features_h
        #H = Features_h_T @ Features_h

        # Check if we should use Nyström approximation
        if use_nystrom:
        
            # Make linear update with Features_h
            self.Y = linear_update(self.Omega, self.Y, 1, ridge, Features_h)
            
            # Get low-rank approximation of G with Lambda <- (Sigma^2 - nu * I)
            U, Lambda = fixed_rank_psd_approximation(self.Omega, self.Y, nystrom_rank)
            
            # In the Nyström approach, this U and Lambda now form a rank-r approximation of G
            # So we have G ≈ U @ Lambda @ U.T
            G_approx = U @ Lambda @ U.T
            
            # Compute W0 using the approximated inverse
            Wo = (G_approx @ self.Q).T
        
        else:
            logging.error("Non-Nyström branch is disabled. Please enable 'use_nystrom'.")
            sys.exit(1)


        self._network.fc.weight.data = Wo[0:self._network.fc.weight.shape[0], :]#.to(self._device)

        # print(self._network.fc.weight.data.shape)
        #
        # U, Sigma, V = torch.linalg.svd(self._network.fc.weight.data)
        # print(Sigma)

        torch.cuda.synchronize()
        self.times['algorithm'] += time.time() - algorithm_start

        self.save_times()
        
        return model

    def setup_RP(self):
        if self.RP_initialized:
            pass
        else:
            M = self.args['M']
            self._network.fc.weight = nn.Parameter(torch.Tensor(self._network.fc.out_features, M).to(self._device)).requires_grad_(False) # num classes in task x M
            self._network.RP_dim = M
            self.W_rand = torch.randn(self._network.fc.in_features, M).to(self._device)
            self._network.W_rand = self.W_rand

            # Initialize sketch with G
            sketch_size = 2 * self.args["nb_classes"]  # Size of the sketch (k parameter)
            self.G = torch.zeros(M, M, dtype=torch.float32, device=self._device)
            self.Omega, self.Y = sketch_initialization(self.G, sketch_size, device=self._device)

            self.Q = torch.zeros(M, self.args["nb_classes"], device=self._device)
            
            # Add a flag to track Y initialization
            self.Y_initialized = False

            self.RP_initialized = True

    def optimise_ridge_parameter(self, Features, Y):
        #ridges = 10.0 ** np.arange(-8, 9)
        ridges = 10.0 ** np.arange(2, 9)
        logging.info(f"Testing with this set of Ridge parameters: {ridges}")
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []
        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]
        for ridge in ridges:
            Wo = torch.linalg.solve(G_val + ridge*torch.eye(G_val.size(dim=0)), Q_val).T #better nmerical stability than .inv
            Y_train_pred = Features[num_val_samples::,:] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples::, :]).item())
        ridge = ridges[np.argmin(np.array(losses))]
        print(losses)
        print('selected lambda =',ridge)
        return ridge
    
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", )
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        self._network.to(self._device)
        
        if self._cur_task == 0:
            # show total parameters and trainable parameters
            total_params = sum(p.numel() for p in self._network.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')
            if total_params != total_trainable_params:
                for name, param in self._network.named_parameters():
                    if param.requires_grad:
                        print(name, param.numel())
            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)

            if self.args['stage1']:
                if self.is_dil:
                    if self.dil_init: # domain-incremental
                        self._init_train(train_loader, test_loader, optimizer, scheduler)

                else: # class-incremental
                    self._init_train(train_loader, test_loader, optimizer, scheduler)

        else:
            pass

        if self.is_dil: # domain-incremental
            if self.dil_init and self.args["use_RP"]:
                self.setup_RP()
        else: # class-incremental
            if self._cur_task == 0 and self.args["use_RP"]:
                self.setup_RP()

        self.replace_fc(train_loader_for_protonet, self._network, None)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)