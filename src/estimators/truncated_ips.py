from typing import Union

import numpy as np

class TruncatedIPS:
    def __init__(self, M: Union[str, float] = "theory", parameter_grid = 30):
        self.M = M
        self.parameter_grid = parameter_grid
    
    def _value_per_sample(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        M = np.sqrt(len(a)) if self.M == "theory" else self.M
        w = np.clip(pi_e(x, a) / pi_0(x, a), 0, M)
        return w * r

    def __call__(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        return np.mean(self._value_per_sample(pi_0, pi_e, x, a, r))

    def variance(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        return np.var(self._value_per_sample(pi_0, pi_e, x, a, r)) / len(x)

    def get_candidate_hyperparameters(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        w = pi_e(x, a) / pi_0(x, a)
        w_min = np.quantile(w, 0.05)
        w_max = np.quantile(w, 0.95)
        theory_value = np.sqrt(len(a))
        space = np.geomspace(w_min, w_max, self.parameter_grid)
        insert_index = np.searchsorted(space, theory_value)
        return {"M": np.insert(space, insert_index, theory_value)[::-1]}
