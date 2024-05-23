from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid


class SlopeTuner:
    def __init__(self, estimator, estimator_kwargs, hyperparameters={}, K=10):
        self.estimator = estimator
        self.estimator_kwargs = estimator_kwargs
        self.hyperparameters = hyperparameters
        self.K = K

    def __call__(self, pi_0, pi_e, x, a, r):
        def confidence_interval(arr):
            """Empirical confidence interval"""
            sorted_arr = sorted(arr)
            return sorted_arr[int(0.025 * len(arr))], sorted_arr[int(0.975 * len(arr))]
        
        chosen_hyperparameter = None
        accumulated_intervals = []
        
        for h in ParameterGrid(self.hyperparameters):
            fulfills_condition = True
            # If the estimator does not have a variance method, we bootstrap the variance
            if not callable(getattr(self.estimator, "variance")):
                h_values = []
                for _ in range(self.K):
                    train_index = np.random.choice(range(len(x)), len(x), replace=True)
                    x_tr, a_tr, r_tr = x[train_index], a[train_index], r[train_index]
                    
                    pi_e_value_hat = self.estimator(**self.estimator_kwargs, **h)(x=x_tr, a=a_tr, r=r_tr,pi_0=pi_0, pi_e=pi_e)
                    h_values.append(pi_e_value_hat)
                interval = confidence_interval(h_values)
            else:
                mean = self.estimator(**self.estimator_kwargs, **h)(x=x, a=a, r=r, pi_0=pi_0, pi_e=pi_e)
                var = self.estimator(**self.estimator_kwargs, **h).variance(x=x, a=a, r=r, pi_0=pi_0, pi_e=pi_e)
                interval = (mean - 2 * np.sqrt(var), mean + 2 * np.sqrt(var))
            for other_interval in accumulated_intervals:
                if not pd.Interval(*interval).overlaps(other_interval):
                    fulfills_condition = False
                    break
            if fulfills_condition:
                chosen_hyperparameter = h
                accumulated_intervals.append(pd.Interval(*interval))
            else:
                break
        
        return self.estimator(**self.estimator_kwargs, **chosen_hyperparameter)(x=x, a=a, r=r, pi_0=pi_0, pi_e=pi_e)
