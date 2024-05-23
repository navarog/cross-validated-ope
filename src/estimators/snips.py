import numpy as np

class SNIPS:
    def _value_per_sample(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        w = pi_e(x, a) / pi_0(x, a)
        w_mean = w.mean()
        w_mean = w_mean if w_mean != 0 else 1
        return w / w_mean * r
    
    def __call__(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        return np.mean(self._value_per_sample(pi_0, pi_e, x, a, r))

    def variance(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        return np.var(self._value_per_sample(pi_0, pi_e, x, a, r)) / len(x)
