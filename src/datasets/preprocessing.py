import numpy as np

def preprocess(X, y, name, **kwargs):
    def ordinal_label(X, y):
        _, y = np.unique(y, return_inverse=True)
        return X, y
    
    def str_to_int_label(X, y):
        y = y.astype(int)
        return X, y
    
    def subsample(X, y, n, replace=False):
        if type(n) == float:
            n = int(n * len(X))
        if n > len(X):
            n = len(X)
        idx = np.random.choice(len(X), n, replace=replace)
        return X[idx], y[idx]
    
    return locals().get(name)(X, y, **kwargs)
    