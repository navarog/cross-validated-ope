from collections import defaultdict

import numpy as np

def sample_from_dataset(X, y, pi, pi_e, n_rounds=None, a_num=None):   
    if n_rounds is None:
        n_rounds = len(X)
    sample_idx = np.random.choice(np.arange(len(X)), size=n_rounds, replace=False)
    sample_x = X[sample_idx]
    sample_y = y[sample_idx]
    pi_train = pi(sample_x)

    sample_a = np.zeros(n_rounds, dtype=int)
    for i in range(n_rounds):
        sample_a[i] = np.random.choice(a_num, p=pi_train[i])
    
    if len(sample_y.shape) == 2:
        sample_r = np.random.binomial(1, sample_y[np.arange(len(sample_y)), sample_a])
        true_value = (pi_e(X) * y).sum(axis=1).mean() 
    else:         
        sample_r = (sample_y == sample_a).astype(int)
        true_value = pi_e(X)[np.arange(len(X)), y].mean()
    return sample_x, sample_a, sample_r, true_value
