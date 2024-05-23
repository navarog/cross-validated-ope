from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

import estimators


class OffPolicyCrossValidationTuner:
    def __init__(self, estimator, estimator_kwargs, hyperparameters={}, K=10, one_standard_error_rule=True, valid_estimator="DR", valid_estimator_kwargs={}, train_valid_ratio="theory"):
        self.estimator = estimator
        self.estimator_kwargs = estimator_kwargs
        self.hyperparameters = hyperparameters
        self.K = K
        self.one_standard_error_rule = one_standard_error_rule
        self.valid_estimator = getattr(estimators, valid_estimator)(**valid_estimator_kwargs)
        self.train_valid_ratio = train_valid_ratio
    
    def __call__(self, pi_0, pi_e, x, a, r):
        def _variance(estimator):
            values = []
            for _ in range(self.K):
                train_idx = np.random.choice(range(len(x)), len(x), replace=True)
                x_tr, a_tr, r_tr = x[train_idx], a[train_idx], r[train_idx]
                values.append(estimator(x=x_tr, a=a_tr, r=r_tr, pi_0=pi_0, pi_e=pi_e))
            return np.var(values)
        errors = defaultdict(list)
        use_bootstrapped_variance = not callable(getattr(self.valid_estimator, "variance")) or not callable(getattr(self.estimator, "variance"))
        if use_bootstrapped_variance:
            valid_variance = _variance(self.valid_estimator) 
        else:
            valid_variance = self.valid_estimator.variance(x=x, a=a, r=r, pi_0=pi_0, pi_e=pi_e)
        
        for h in ParameterGrid(self.hyperparameters):
            if self.train_valid_ratio != "theory":
                train_valid_ratio = self.train_valid_ratio
            else:
                if use_bootstrapped_variance:
                    h_variance = _variance(self.estimator(**self.estimator_kwargs, **h))
                else:
                    h_variance = self.estimator(**self.estimator_kwargs, **h).variance(x=x, a=a, r=r, pi_0=pi_0, pi_e=pi_e)
                train_valid_ratio = h_variance / (h_variance + valid_variance) if (h_variance + valid_variance) != 0 else 0.5
            for _ in range(self.K):
                train_idx = np.random.choice(range(len(x)), max(int(len(x) * train_valid_ratio), 1), replace=False)
                train_mask = np.zeros(len(x), dtype=bool)
                train_mask[train_idx] = True
                x_tr, a_tr, r_tr = x[train_mask], a[train_mask], r[train_mask]
                x_valid, a_valid, r_valid = x[~train_mask], a[~train_mask], r[~train_mask]
                pi_e_value = self.valid_estimator(x=x_valid, a=a_valid, r=r_valid, pi_0=pi_0, pi_e=pi_e)
            
                pi_e_value_hat = self.estimator(**self.estimator_kwargs, **h)(x=x_tr, a=a_tr, r=r_tr, pi_0=pi_0, pi_e=pi_e)
                errors[str(h)].append((pi_e_value_hat - pi_e_value)**2)
        
        result = pd.DataFrame(errors)
        if self.one_standard_error_rule:
            mean_upper_bound = result.mean() + result.std() / result.count().map(np.sqrt)
            chosen_hyperparameter = mean_upper_bound.idxmin()
        else:
            chosen_hyperparameter = result.mean().idxmin() 
        
        return self.estimator(**self.estimator_kwargs, **eval(chosen_hyperparameter))(x=x, a=a, r=r, pi_0=pi_0, pi_e=pi_e)
