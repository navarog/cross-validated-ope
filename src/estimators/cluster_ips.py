import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .reward_model import RewardModel

class GroupIPS:
    def __init__(self, K="theory", cluster_method="bins", only_actions = False, reward_model = {}):
        self.K = K
        self.cluster_method = cluster_method
        self.only_actions = only_actions
        self.model = RewardModel(**reward_model)
        
    def _value_per_sample(self, pi_0, pi_e, x, a, r):
        def cluster_ips(clusters, pi_0, pi_e, x, a, r):
            num_clusters = clusters.max() + 1
            pi_e_clusters = np.zeros((len(x), num_clusters))
            pi_0_clusters = np.zeros((len(x), num_clusters))
            for cluster in np.unique(clusters):
                cluster_mask = clusters == cluster
                pi_e_clusters[:, cluster] = (pi_e * cluster_mask).sum(axis=1)
                pi_0_clusters[:, cluster] = (pi_0 * cluster_mask).sum(axis=1)

            w = pi_e_clusters[np.arange(len(a)), clusters[np.arange(len(a)), a]] / pi_0_clusters[np.arange(len(a)), clusters[np.arange(len(a)), a]]
            return w * r
        
        pi_0 = pi_0(x)
        pi_e = pi_e(x)
        K = int(np.sqrt(pi_0.shape[1])) if self.K == "theory" else self.K
        self.model.fit(x, a, r, a_num=pi_0.shape[1])
        hat_r = self.model.predict(x)
        if self.only_actions == True:
            hat_r = np.tile(hat_r.mean(axis=0), len(x)).reshape(len(x), -1)
            
        if self.cluster_method == "uniform":
            clusters = np.array([pd.qcut(row, K, labels=False, duplicates="drop") if row.std() != 0 else np.zeros_like(row, dtype=int) for row in hat_r])
        elif self.cluster_method == "bins":
            eps = 1e-7
            intervals = np.linspace(hat_r.min() - eps, hat_r.max() + eps, K + 1)
            clusters = np.digitize(hat_r, intervals) - 1            
        elif self.cluster_method == "kmeans":
            clusters = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(hat_r.reshape(-1, 1)).labels_.reshape(len(x), -1)
        else:
            raise NotImplementedError(f"Cluster method {self.cluster_method} not implemented")
        return cluster_ips(clusters, pi_0, pi_e, x, a, r)        
    
    def __call__(self, pi_0, pi_e, x, a, r):
        return self._value_per_sample(pi_0, pi_e, x, a, r).mean()

    def variance(self, pi_0, pi_e, x, a, r):
        return np.var(self._value_per_sample(pi_0, pi_e, x, a, r)) / len(x)
