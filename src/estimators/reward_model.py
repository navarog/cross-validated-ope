import copy
import sys
from functools import reduce

import numpy as np
import sklearn


class RewardModel:
    cached_models = {}
    
    def __init__(
        self,
        model_class="sklearn.linear_model.Ridge",
        model_kwargs={"alpha": 1e-3, "solver": "cholesky", "fit_intercept": True},
    ):
        self.model_class = reduce(getattr, model_class.split("."), sys.modules[__name__])
        self.model_kwargs = model_kwargs
        self.is_sklearn = model_class.startswith("sklearn")

    def fit(self, x_train, a_train, r_train, a_num=None):
        self.a_num = a_num if a_num is not None else a_train.max() + 1
        hashed_data = hash(x_train.tostring() + a_train.tostring() + r_train.tostring() + bytes(str(self.model_class), encoding='utf8') + bytes(str(self.model_kwargs), encoding='utf8'))
        if hashed_data in RewardModel.cached_models:
            self.__dict__.update(RewardModel.cached_models[hashed_data].__dict__)
            return self
        
        if self.is_sklearn:
            self.models = []
            for a in range(self.a_num):
                a_idx = a_train == a

                if a_idx.sum() == 0:
                    self.models.append(None)
                    continue
                
                unique_r = np.unique(r_train[a_idx])
                if len(unique_r) == 1:
                    self.models.append(unique_r[0])
                    continue

                self.models.append(self.model_class(**self.model_kwargs).fit(x_train[a_idx], r_train[a_idx]))
        else:
            self.model = self.model_class(**self.model_kwargs).fit(x_train, a_train, r_train, self.a_num)
        RewardModel.cached_models[hashed_data] = self
        return self

    def predict(self, X, a=None):
        if self.is_sklearn:
            predictions = np.zeros((len(X), self.a_num))
            iterate = np.unique(a) if a is not None else range(self.a_num)
            for a in iterate:
                if np.isscalar(self.models[a]):
                    predictions[:, a] = self.models[a]
                elif self.models[a] is not None:
                    predictions[:, a] = self.models[a].predict_proba(X)[:, 0] if hasattr(self.models[a], "predict_proba") else self.models[a].predict(X)
            return predictions
        else:
            return self.model.predict(X, a)

    def clear_cache():
        RewardModel.cached_models = {}
