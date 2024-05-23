import numpy as np
from scipy.special import expit

def get_dataset(n = 30000, d = 10, a_num = 100, n_clusters = 20):
    X = np.random.normal(size=(n, d))
    theta_cluster = np.random.uniform(-1, 1, size=(n_clusters, d)) * 3
    action_cluster = np.random.choice(n_clusters, a_num)
    np.random.shuffle(action_cluster)
    theta_action = np.random.uniform(-1, 1, size=(a_num, d)) * 1 + theta_cluster[action_cluster]
    y = expit(np.dot(X, theta_action.T) - 2)
    
    return X, y