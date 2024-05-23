import estimators

class EstimatorSelection:
    def __init__(self, estimator = "DM", hyperparameters = [], **estimator_kwargs):
        self.estimator = getattr(estimators, estimator)(**estimator_kwargs)
        self.hyperparameters = hyperparameters
    
    def __call__(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        return self.estimator(pi_0=pi_0, pi_e=pi_e, x=x, a=a, r=r)

    def variance(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        return self.estimator.variance(pi_0=pi_0, pi_e=pi_e, x=x, a=a, r=r)
    
    def get_candidate_hyperparameters(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        candidates = []
        for h in self.hyperparameters:
            estimator = h["name"]
            estimator_kwargs = h["kwargs"] if "kwargs" in h else {}
            hyperparameters = h["hyperparameters"] if "hyperparameters" in h else {}
            if hyperparameters == "dynamic":
                hyperparameters = getattr(estimators, estimator)(**estimator_kwargs).get_candidate_hyperparameters(pi_0=pi_0, pi_e=pi_e, x=x, a=a, r=r)
            candidate = {"estimator": [estimator]}
            candidate.update(**hyperparameters)
            for key, value in estimator_kwargs.items():
                candidate[key] = [value]
            candidates.append(candidate)
        return candidates