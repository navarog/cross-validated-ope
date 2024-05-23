import numpy as np

from .reward_model import RewardModel

class DR:
    def __init__(self, reward_model = {}):
        self.model = RewardModel(**reward_model)
    
    def _value_per_sample(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        pi_0_x = pi_0(x)
        pi_e_x = pi_e(x)
        w = pi_e_x[np.arange(len(a)), a] / pi_0_x[np.arange(len(a)), a]
        
        self.model.fit(x, a, r, a_num=pi_e_x.shape[1])
        hat_r_a = self.model.predict(x)
        dm_predictions = (hat_r_a * pi_e_x).sum(axis=1)
        
        return dm_predictions + w * (r - hat_r_a[np.arange(len(a)), a])
        
    def __call__(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        return np.mean(self._value_per_sample(pi_0, pi_e, x, a, r))
    
    def variance(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        return np.var(self._value_per_sample(pi_0, pi_e, x, a, r)) / len(x)
