### Policy Adaptive Estimator Selection for Off-Policy Evaluation
### This code is adapted from https://github.com/sony/ds-research-code/tree/master/aaai23-pasif

from collections import defaultdict

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from sklearn.model_selection import ParameterGrid
from torch.nn import functional as F
from torch.utils.data import DataLoader

from utils.policy import FixedPolicy


class PASIFTuner:
    def __init__(
        self,
        estimator,
        estimator_kwargs,
        hyperparameters={},
        K=10,
        regularization_weight=1e3,
        k=0.2,
        k_epsilon=2e-2,
        n_epochs=5000,
        lr=0.001,
        batch_size=1000,
        early_stopping_rounds=5,
        early_stopping_tolerance=1e-3,
        w_clip_value=1e7,
    ):
        self.estimator = estimator
        self.estimator_kwargs = estimator_kwargs
        self.hyperparameters = hyperparameters
        self.K = K
        self.regularization_weight = regularization_weight
        self.k = k
        self.k_epsilon = k_epsilon
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_tolerance = early_stopping_tolerance
        self.w_clip_value = w_clip_value

    def __call__(self, pi_0, pi_e, x, a, r):
        errors = defaultdict(list)

        for _ in tqdm.trange(self.K, leave=False, desc="Bootstrap run"):
            idx = np.random.choice(len(x), len(x), replace=True)
            x_boot, a_boot, r_boot = x[idx], a[idx], r[idx]

            surrogate_pi_0, surrogate_pi_e, x_tr, a_tr, r_tr, x_val, a_val, r_val = self.fit_and_split(
                pi_0, pi_e, x_boot, a_boot, r_boot
            )
            
            if len(r_val) == 0 or len(r_tr) == 0:
                continue
            
            pi_e_value = r_val.mean()

            for h in ParameterGrid(self.hyperparameters):
                pi_e_value_hat = self.estimator(**self.estimator_kwargs, **h)(
                    x=x_tr, a=a_tr, r=r_tr, pi_0=surrogate_pi_0, pi_e=surrogate_pi_e
                )
                errors[str(h)].append((pi_e_value_hat - pi_e_value) ** 2)

        result = pd.DataFrame(errors)
        chosen_hyperparameter = result.mean().idxmin()

        return self.estimator(**self.estimator_kwargs, **eval(chosen_hyperparameter))(
            x=x, a=a, r=r, pi_0=pi_0, pi_e=pi_e
        )

    def fit_and_split(self, pi_0, pi_e, x, a, r):
        pi_0 = pi_0(x)
        pi_e = pi_e(x)
        n_actions = pi_0.shape[1]

        class SubsamplingRuleLoss(nn.Module):
            def __init__(self, k, k_epsilon, regularization_weight):
                super().__init__()
                self.k = k
                self.k_epsilon = k_epsilon
                self.regularization_weight = regularization_weight
                self.loss_r = 0.0
                self.loss_d = 0.0
                self.loss = 0.0
                self.samples = 0

            def forward(self, outputs, targets):
                # outputs: (batch_size, n_actions + 1), outputs with actual action and fixed actions
                # targets: (batch_size, n_actions + 2), w, pi_b, pscore
                marginal_p = torch.zeros(len(targets)).to(outputs.device)
                for action in range(1, 1 + n_actions):
                    marginal_p += targets[:, action] * outputs[:, action]
                tilde_p_e = targets[:, -1] * (outputs[:, 0] / marginal_p)
                tilde_p_b = targets[:, -1] * ((1.0 - outputs[:, 0]) / (1.0 - marginal_p))
                tilde_w = tilde_p_e / tilde_p_b  # importance weight by NN
                loss_d = ((tilde_w - targets[:, 0]) ** 2).mean()
                loss_r = (((marginal_p - self.k) ** 2) * (torch.abs(marginal_p - self.k) > self.k_epsilon)).mean()
                loss = loss_d + self.regularization_weight * loss_r
                self.loss_r += loss_r.sum().item()
                self.loss_d += loss_d.sum().item()
                self.loss += loss.sum().item()
                self.samples += len(targets)
                return loss
            
            def mean_loss(self):
                return self.loss / self.samples
            
            def zero_loss(self):
                self.loss_r = 0.0
                self.loss_d = 0.0
                self.loss = 0.0
                self.samples = 0

        subsampling_loss = SubsamplingRuleLoss(self.k, self.k_epsilon, self.regularization_weight)

        # dim of context + action
        dim_context = x.shape[1] + 1

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(dim_context, 100)
                self.bn1 = nn.BatchNorm1d(100)
                self.fc2 = nn.Linear(100, 100)
                self.bn2 = nn.BatchNorm1d(100)
                self.fc3 = nn.Linear(100, 1)

            def forward(self, x):
                x = self.fc1(x)
                x = self.bn1(x)
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
                x = self.fc2(x)
                x = self.bn2(x)
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
                x = self.fc3(x)
                x = torch.sigmoid(x)
                return x

        torch.manual_seed(np.random.get_state()[1][0])  # set the same seed as in numpy
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        x_and_a = np.concatenate([x, a[:, None]], axis=-1)
        x_and_a = torch.from_numpy(x_and_a).float() # data of context and action

        w = np.clip(
            pi_e[np.arange(len(x)), a] / pi_0[np.arange(len(x)), a],
            0.0,
            self.w_clip_value,    
        )
        w = torch.from_numpy(w).float() # ideal importance weight

        # create tensor data set (context_and_actual_action, importance_weight, behavior_policy, propensity_score)
        tensor_dataset = torch.utils.data.TensorDataset(
            x_and_a, w, torch.from_numpy(pi_0), torch.from_numpy(pi_e[np.arange(len(x)), a])
        )
        dataloader = DataLoader(tensor_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        self.net = Net().to(device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        progress_bar = tqdm.trange(self.n_epochs, leave=False)
        min_loss = float("inf")
        early_stopping_counter = 0
        best_net = copy.deepcopy(self.net)
        for _ in progress_bar:
            subsampling_loss.zero_loss()
            for t_x_and_a, t_w, t_pi_b, t_p in dataloader:
                t_x_and_a, t_w, t_pi_b, t_p = (
                    t_x_and_a.to(device),
                    t_w.to(device),
                    t_pi_b.to(device),
                    t_p.to(device),
                )

                optimizer.zero_grad()

                output_actual = self.net(t_x_and_a)
                output_list = [output_actual]
                for action in range(n_actions):
                    t_x_and_fixed_action = copy.deepcopy(t_x_and_a)
                    t_x_and_fixed_action[:, -1] = action
                    output_fixed_action = self.net(t_x_and_fixed_action)
                    output_list.append(output_fixed_action)
                outputs = torch.cat(output_list, axis=1)
                targets = torch.cat([t_w[:, None], t_pi_b, t_p[:, None]], axis=1)

                loss = subsampling_loss(outputs=outputs, targets=targets)
                loss.backward()
                optimizer.step()

                progress_bar.set_description(f"Subsampling loss: {subsampling_loss.mean_loss():.4f} (r: {subsampling_loss.loss_r / subsampling_loss.samples:.4f}, d: {subsampling_loss.loss_d / subsampling_loss.samples:.4f})")
            
            if subsampling_loss.mean_loss() < min_loss - self.early_stopping_tolerance:
                min_loss = subsampling_loss.mean_loss()
                best_net = copy.deepcopy(self.net)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= self.early_stopping_rounds:
                    break
                
        self.net = best_net
        self.net.eval()
        with torch.no_grad():
            x_and_a = x_and_a.to(device)
            sampling_probability = self.net(x_and_a)
            sampling_probability = sampling_probability.cpu().detach().numpy().flatten()

            marginal_p = np.zeros(len(x))
            output_fixed_action = np.zeros((len(x), n_actions))
            for action in range(n_actions):
                x_and_a[:, -1] = action
                fixed_action_probability = self.net(x_and_a).cpu().detach().numpy().flatten()
                output_fixed_action[:, action] = fixed_action_probability
                marginal_p += pi_0[np.arange(len(x)), action] * fixed_action_probability

            marginal_p = np.clip(marginal_p, 1e-5, 1 - 1e-5)
            surrogate_pi_e = pi_e * output_fixed_action / marginal_p[:, None]
            surrogate_pi_0 = pi_0 * (1 - output_fixed_action) / (1 - marginal_p[:, None])

            train_mask = np.random.rand(len(x)) > sampling_probability
            x_val, a_val, r_val = x[~train_mask], a[~train_mask], r[~train_mask]
            x_tr, a_tr, r_tr = x[train_mask], a[train_mask], r[train_mask]

            return (
                FixedPolicy(x=x, pi=surrogate_pi_0),
                FixedPolicy(x=x, pi=surrogate_pi_e),
                x_tr,
                a_tr,
                r_tr,
                x_val,
                a_val,
                r_val,
            )
