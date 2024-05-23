import numpy as np

from .reward_model import RewardModel
        
        
class DM:
    def __init__(self, reward_model = {}):
        self.model = RewardModel(**reward_model)
    
    def _value_per_sample(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        pi_e = pi_e(x)
        self.model.fit(x, a, r, a_num=pi_e.shape[1])
        return (self.model.predict(x) * pi_e).sum(axis=1)
        
    def __call__(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        return np.mean(self._value_per_sample(pi_0, pi_e, x, a, r))
    
    def variance(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        return np.var(self._value_per_sample(pi_0, pi_e, x, a, r)) / len(x)
