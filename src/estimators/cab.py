import numpy as np

from .reward_model import RewardModel

class CAB:
    def __init__(self, M = 1, parameter_grid = 30, reward_model = {}):
        self.model = RewardModel(**reward_model)
        self.M = M
        self.parameter_grid = parameter_grid
    
    def _value_per_sample(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        pi_0 = pi_0(x)
        pi_e = pi_e(x)
        w_alpha = 1 - np.clip(self.M * pi_0 / pi_e, 1, np.inf)
        w_beta = np.clip(self.M * pi_0[np.arange(len(a)), a] / pi_e[np.arange(len(a)), a], 1, np.inf)
        beta = r / pi_0[np.arange(len(a)), a]
        self.model.fit(x, a, r, a_num=pi_e.shape[1])
        alpha = self.model.predict(x)
        dm_result = (pi_e * w_alpha * alpha ).sum(axis=1)
        ips_result = (pi_e[np.arange(len(a)), a] * w_beta * beta)
        return dm_result + ips_result
        
    def __call__(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        return np.mean(self._value_per_sample(pi_0, pi_e, x, a, r))
        
    def variance(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        return np.var(self._value_per_sample(pi_0, pi_e, x, a, r)) / len(x)
    
    def get_hat_w(self, w):
        return self.lambda_ / (self.lambda_ + w ** 2) * w
    
    def get_candidate_hyperparameters(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        w = pi_e(x, a) / pi_0(x, a)
        w_min = np.quantile(w, 0.05)
        w_max = np.quantile(w, 0.95)
        return {"M": np.geomspace(w_min, w_max, self.parameter_grid)[::-1]}
