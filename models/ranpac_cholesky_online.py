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
from utils.toolkit import target2onehot, tensor2numpy

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
        logging.info('test 1 time')
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
        logging.info('test 2 time')

    import torch

    def cholesky_update(L, x, add=True):
        """
        Update the Cholesky decomposition L of matrix A, where A = L @ L.T

        When adding a rank-1 update: A' = A + x @ x.T (if add=True)
        When subtracting a rank-1 update: A' = A - x @ x.T (if add=False)

        Returns the updated Cholesky factor L'

        Parameters:
        -----------
        L : torch.Tensor
        Lower triangular Cholesky factor of shape (n, n)
        x : torch.Tensor
        Vector for rank-1 update of shape (n,)
        add : bool
        If True, perform rank-1 update; if False, perform rank-1 downdate

        Returns:
        --------
        L_new : torch.Tensor
        Updated Cholesky factor
        """
        if not add:
            # For downdate, we need to ensure the result will still be positive definite
            v = torch.triangular_solve(x.unsqueeze(1), L, upper=False)[0].squeeze()
            v_norm_sq = torch.sum(v**2)
        if v_norm_sq >= 1.0 and not add:
            raise ValueError("Cholesky downdate would result in a non-positive definite matrix")

        n = L.shape[0]
        L_new = L.clone()

        if add:
            # Rank-1 update: A + x @ x.T
            for k in range(n):
                # Compute the rank-1 update for the k-th row
                r = torch.sqrt(L_new[k, k]**2 + (x[k]**2 if add else -x[k]**2))
                c = r / L_new[k, k]
                s = x[k] / L_new[k, k]
                L_new[k, k] = r

                if k < n - 1:
                    # Update the remaining elements in the k-th column
                    L_new[k+1:, k] = (L_new[k+1:, k] + s * x[k+1:]) / c
                    # Update x for the next iteration
                    x[k+1:] = x[k+1:] - s * L_new[k+1:, k]
        else:
            # Rank-1 downdate: A - x @ x.T
            for k in range(n):
                # Solve the linear system to get the effect on this column
                p = x[k] / L_new[k, k]
                r = torch.sqrt(L_new[k, k]**2 - x[k]**2)
                c = r / L_new[k, k]
                s = p / L_new[k, k]
                L_new[k, k] = r

                if k < n - 1:
                    # Update the remaining rows
                    x[k+1:] = x[k+1:] - p * L_new[k+1:, k]
                    L_new[k+1:, k] = c * L_new[k+1:, k] - s * x[k+1:]

        return L_new


    def cholesky_update_batch(L, X, add=True):
        """
        Update the Cholesky decomposition L for multiple rank-1 updates at once.
        This is more efficient than applying individual updates sequentially.

        Parameters:
        -----------
        L : torch.Tensor
        Lower triangular Cholesky factor of shape (n, n)
        X : torch.Tensor
        Matrix of shape (n, m) where each column is a rank-1 update vector
        add : bool
        If True, perform rank-1 updates; if False, perform rank-1 downdates

        Returns:
        --------
        L_new : torch.Tensor
        Updated Cholesky factor after all updates
        """
        L_current = L.clone()
        for i in range(X.shape[1]):
            L_current = cholesky_update(L_current, X[:, i], add=add)

        return L_current


    def test_cholesky_update():
        """
        Test function to verify the correctness of the implementation
        """
        # Create a positive definite matrix A = B @ B.T + diagonal for stability
        n = 5
        torch.manual_seed(0)
        B = torch.randn(n, n)
        A = B @ B.T + torch.eye(n) * 0.1

        # Compute original Cholesky decomposition
        L_original = torch.linalg.cholesky(A)

        # Create a rank-1 update vector
        x = torch.randn(n)

        # Create the updated matrix directly
        A_updated = A + torch.outer(x, x)
        L_true = torch.linalg.cholesky(A_updated)

        # Update using our function
        L_updated = cholesky_update(L_original, x, add=True)

        # Compare the results
        diff = torch.norm(L_updated - L_true)
        print(f"Difference between direct computation and update: {diff:.8f}")

        # Test downdate
        A_downdated = A - torch.outer(x, x)
        L_downdated_true = torch.linalg.cholesky(A_downdated)
        L_downdated = cholesky_update(L_original, x, add=False)
        diff_down = torch.norm(L_downdated - L_downdated_true)
        print(f"Difference for downdate: {diff_down:.8f}")


    def replace_fc(self, trainloader, model, args):       
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

                #algorithm_start = time.time()
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
                #self.times['algorithm'] += time.time() - algorithm_start

        algorithm_start = time.time()

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        Y = target2onehot(label_list, self.args["nb_classes"])
        Features_h = F.relu(embedding_list @ self.W_rand.cpu())
        self.Q = self.Q + Features_h.T @ Y
        self.G = self.G + Features_h.T @ Features_h

        #self.L, self.D, perm = torch.linalg.ldl_factor(self.G)
        #self.L, self.D = torch.linalg.ldl_factor(self.G, hermitian=True)
        #self.D = self.D.to(dtype=self.G.dtype)
        #self.G = self.L @ self.D @ self.L.T

        '''
        if self.args['search_ridge']:
            ridge = self.optimise_ridge_parameter(Features_h, Y)
        else:
            ridge = self.args['ridge']
        '''
        ridge = 100000

        W_aux = self.G + ridge*torch.eye(self.G.size(dim=0))

        if torch.all(self.L == 0):  # Check if L is uninitialized
            self.L = torch.linalg.cholesky(W_aux)

        else:
            self.L = cholesky_update_batch(self.L, Features_h.T, add=True)

        #self.L = torch.linalg.cholesky(W_aux)
        Wo = torch.cholesky_solve(self.Q, self.L).T

        #Wo = torch.linalg.solve(self.G + ridge*torch.eye(self.G.size(dim=0)), self.Q).T # better nmerical stability than .invv
        self._network.fc.weight.data = Wo[0:self._network.fc.weight.shape[0], :].to(self._device)

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

            self.Q = torch.zeros(M, self.args["nb_classes"])
            self.G = torch.zeros(M, M, dtype=torch.float32)
            self.L = torch.zeros(M, M, dtype=torch.float32)
            #self.D = torch.zeros(M, M, dtype=torch.float32)

            self.RP_initialized = True

    def optimise_ridge_parameter(self, Features, Y):
        ridges = 10.0 ** np.arange(-8, 9)
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []
        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]



        #L_val, D_val, perm = torch.linalg.ldl_factor(G_val)
        #L_val, D_val = torch.linalg.ldl_factor(G_val, hermitian=True)

        #D_val = D_val.to(dtype=self.G.dtype)

        #L_val = torch.linalg.cholesky(G_val)
        #G_val = L_val @ D_val @ L_val.T

        for ridge in ridges:

            W_aux_val = G_val + ridge*torch.eye(G_val.size(dim=0)).T

            L_val = torch.linalg.cholesky(W_aux_val)

            Wo = torch.cholesky_solve(Q_val, L_val)

            #Wo = torch.linalg.solve(G_val + ridge*torch.eye(G_val.size(dim=0)), Q_val).T #better nmerical stability than .inv
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
        self.save_times() # Included by Cristobal

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