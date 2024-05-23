import numpy as np

from .reward_model import RewardModel

class DRos:
    def __init__(self, lambda_ = 1, parameter_grid = 30, reward_model = {}):
        self.model = RewardModel(**reward_model)
        self.lambda_ = lambda_
        self.parameter_grid = parameter_grid
        
    def _value_per_sample(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        pi_0_x = pi_0(x)
        pi_e_x = pi_e(x)
        w = pi_e_x[np.arange(len(a)), a] / pi_0_x[np.arange(len(a)), a]
        hat_w = self.get_hat_w(w)
        
        self.model.fit(x, a, r, a_num=pi_e_x.shape[1])
        hat_r_a = self.model.predict(x)
        dm_predictions = (hat_r_a * pi_e_x).sum(axis=1)
        
        return dm_predictions + hat_w * (r - hat_r_a[np.arange(len(a)), a])
        
    def __call__(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        return np.mean(self._value_per_sample(pi_0, pi_e, x, a, r))
    
    def variance(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        return np.var(self._value_per_sample(pi_0, pi_e, x, a, r)) / len(x)

    def get_hat_w(self, w):
        return self.lambda_ / (self.lambda_ + w ** 2) * w
    
    def get_candidate_hyperparameters(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        w = (pi_e(x, a) / pi_0(x, a)) ** 2
        w_min = np.quantile(w, 0.05) * 0.01
        w_max = np.quantile(w, 0.95) * 100
        return {"lambda_": np.geomspace(w_min, w_max, self.parameter_grid)[::-1]}
    

class DRps(DRos):
    def get_hat_w(self, w):
        return np.clip(w, a_min=-np.inf, a_max=self.lambda_)
    
    def get_candidate_hyperparameters(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        w = pi_e(x, a) / pi_0(x, a)
        w_min = np.quantile(w, 0.05)
        w_max = np.quantile(w, 0.95)
        return {"lambda_": np.geomspace(w_min, w_max, self.parameter_grid)[::-1]}    
