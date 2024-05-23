import numpy as np
from sklearn.model_selection import ParameterGrid


class DRsTuner:
    def __init__(self, estimator, estimator_kwargs, hyperparameters={}):
        self.estimator = estimator
        self.estimator_kwargs = estimator_kwargs
        self.hyperparameters = hyperparameters
    
    def __call__(self, pi_0, pi_e, x, a, r):
        errors = {}
        
        w = pi_e(x, a) / pi_0(x, a)
        
        for h in ParameterGrid(self.hyperparameters):
            estimator = self.estimator(**self.estimator_kwargs, **h)
            variance = estimator.variance(x=x, a=a, r=r, pi_0=pi_0, pi_e=pi_e)
            hat_w = estimator.get_hat_w(w)
            hat_r = estimator.model.predict(x)
            
            bias = ((hat_w - w) * (r - hat_r[np.arange(len(a)), a])).mean()
            errors[str(h)] = variance + bias ** 2
        
        chosen_hyperparameter = min(errors, key=errors.get) 
        return self.estimator(**self.estimator_kwargs, **eval(chosen_hyperparameter))(x=x, a=a, r=r, pi_0=pi_0, pi_e=pi_e)
