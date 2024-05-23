import numpy as np
from scipy.special import expit

class SubGaussianIPS:
    def __init__(self, lambda_ = "theory", parameter_grid = 30, confidence_level = 0.95):
        self.lambda_ = lambda_
        self.parameter_grid = parameter_grid
        self.confidence_level = confidence_level
        
    def _value_per_sample(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        w = pi_e(x, a) / pi_0(x, a)
        lambda_ = self._tune_lambda(w) if self.lambda_ == "theory" else self.lambda_
        w_lambda = w / (1 - lambda_ + lambda_ * w)
        return w_lambda * r
    
    def __call__(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        return np.mean(self._value_per_sample(pi_0, pi_e, x, a, r))

    def variance(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        return np.var(self._value_per_sample(pi_0, pi_e, x, a, r)) / len(x)

    def get_candidate_hyperparameters(self, pi_0=None, pi_e=None, x=None, a=None, r=None):
        space = expit(np.linspace(-10, 10, self.parameter_grid))
        theory_value = self._tune_lambda(pi_e(x, a) / pi_0(x, a))
        insert_index = np.searchsorted(space, theory_value)
        return {"lambda_": np.insert(space, insert_index, theory_value)}
    
    def _tune_lambda(self, w):
        def error(lambda_):
            moment = len(w) ** 0.25
            w_lambda = ((lambda_ + (1 - lambda_) * w ** moment) ** (2 / moment))
            return w_lambda.mean() * lambda_ ** 2 - (2 * np.log(1 / self.confidence_level) / (3 * len(w_lambda)))
        
        lambda_ = 0.5
        learning_rate = 0.1
        epsilon = 1e-5
        tolerance = 1e-6
        max_iterations = 1000
        iteration = 0

        while True:
            err = error(lambda_)
            
            # Calculate the diff
            err_d = (err - error(lambda_ - epsilon)) / epsilon

            # Compute the new error
            lambda_new = np.clip(lambda_ - learning_rate * err / err_d, 0, 1)
                            
            err_new = error(lambda_new)
            # Check for convergence
            if abs(err_new - err) < tolerance:
                break

            # Update lambda and error
            lambda_ = lambda_new
            err = err_new

            # Check for max iterations
            iteration += 1
            if iteration == max_iterations:
                break

        return lambda_         
