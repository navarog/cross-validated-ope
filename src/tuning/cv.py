from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, KFold
import estimators


class CrossValidationTuner:
    def __init__(self, estimator, estimator_kwargs, hyperparameters={}, k_folds=10, one_standard_error_rule=True, valid_estimator="DR", valid_estimator_kwargs={}):
        self.estimator = estimator
        self.estimator_kwargs = estimator_kwargs
        self.hyperparameters = hyperparameters
        self.k_folds = k_folds
        self.one_standard_error_rule = one_standard_error_rule
        self.valid_estimator = getattr(estimators, valid_estimator)(**valid_estimator_kwargs)

    def __call__(self, pi_0, pi_e, x, a, r):
        errors = defaultdict(list)
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=0)
        for train_index, valid_index in kf.split(x, a):
            x_tr, a_tr, r_tr = x[train_index], a[train_index], r[train_index]
            x_valid, a_valid, r_valid = x[valid_index], a[valid_index], r[valid_index]
            pi_e_value = self.valid_estimator(x=x_valid, a=a_valid, r=r_valid, pi_0=pi_0, pi_e=pi_e)
            
            for h in ParameterGrid(self.hyperparameters):
                pi_e_value_hat = self.estimator(**self.estimator_kwargs, **h)(x=x_tr, a=a_tr, r=r_tr, pi_0=pi_0, pi_e=pi_e)
                errors[str(h)].append((pi_e_value_hat - pi_e_value)**2)
        
        result = pd.DataFrame(errors)
        if self.one_standard_error_rule:
            mean_upper_bound = result.mean() + result.std() / result.count().map(np.sqrt)
            chosen_hyperparameter = mean_upper_bound.idxmin()
        else:
            chosen_hyperparameter = result.mean().idxmin() 

        return self.estimator(**self.estimator_kwargs, **eval(chosen_hyperparameter))(x=x, a=a, r=r, pi_0=pi_0, pi_e=pi_e)
