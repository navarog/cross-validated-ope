import sys
import warnings
from functools import reduce

import numpy as np
import sklearn.linear_model
import sklearn.neural_network
import sklearn.tree
import sklearn.ensemble
import sklearn.exceptions
from scipy.special import softmax

from datasets.preprocessing import preprocess


warnings.simplefilter(action='ignore', category=sklearn.exceptions.ConvergenceWarning)


class BasePolicy:
    def __init__(self, a_num=100):
        self.a_num = a_num
    
    def fit(self, X, y):
        return self
    
    def __call__(self, X, a=None):
        pass
    
    
class UniformPolicy(BasePolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, X, a=None):
        probabilities = np.tile(np.ones(self.a_num) / self.a_num, (len(X), 1))
        return probabilities if a is None else probabilities[np.arange(len(X)), a]


class RandomSoftmaxPolicy(BasePolicy):
    def __init__(self, temperature=1, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
    
    def __call__(self, X, a=None):
        logits = X @ self.theta
        probabilities = softmax(logits * self.temperature, axis=1)
        return probabilities if a is None else probabilities[np.arange(len(X)), a]
    
    def fit(self, X, y):
        self.theta = np.random.randn(X.shape[1], self.a_num)
        return self


class FixedPolicy(BasePolicy):
    def __init__(self, x=None, pi=None, **kwargs):
        super().__init__(**kwargs)
        self.a_num = pi.shape[1]
        self.pi = dict((tuple(x), p) for x, p in zip(x, pi))
        
    def __call__(self, X, a=None):
        probabilities = np.array([self.pi.get(tuple(x)) for x in X])
        return probabilities if a is None else probabilities[np.arange(len(X)), a]

class SoftmaxPolicy(BasePolicy):
    def __init__(self, temperature=1, model_class = "sklearn.linear_model.LogisticRegression", model_kwargs = {}, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.model = reduce(getattr, model_class.split("."), sys.modules[__name__])(**model_kwargs)
        self.is_sklearn = model_class.startswith("sklearn")
    
    def fit(self, X, y):
        if len(y.shape) == 1:
            dummy_X = np.zeros((self.a_num, *X.shape[1:]))
            dummy_y = np.arange((self.a_num))
            X = np.concatenate([dummy_X, X])
            y = np.concatenate([dummy_y, y])
        
        self.model.fit(X, y)
        return self
    
    def __call__(self, X, a=None):
        if self.temperature is None:
            predictions = self.model.predict(X)
            predictions -= np.clip(predictions.min(axis=1, keepdims=True), a_min=None, a_max=0)
            probabilities = predictions / predictions.sum(axis=1, keepdims=True)
        else:
            predictions = self.model.predict_proba(X) if hasattr(self.model, "predict_proba") else self.model.predict(X)
            probabilities = softmax(predictions * self.temperature, axis=1)
        return probabilities if a is None else probabilities[np.arange(len(X)), a]
    

def get_policy(name = "uniform", X=None, y=None, a_num=10000, preprocessing = [], **kwargs):
    for preprocessing_step in preprocessing:
        preprocessing_kwargs = preprocessing_step["kwargs"] if "kwargs" in preprocessing_step else {}
        X, y = preprocess(X, y, preprocessing_step["name"], **preprocessing_kwargs)
    if name == "softmax":
        return SoftmaxPolicy(a_num=a_num, **kwargs).fit(X, y)
    if name == "uniform":
        return UniformPolicy(a_num=a_num)
    if name == "random":
        return RandomSoftmaxPolicy(a_num=a_num, **kwargs).fit(X, y)
    
    raise NotImplementedError("Unknown policy name")
