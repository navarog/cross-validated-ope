import time

import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import ParameterGrid

import estimators
import estimators.reward_model
import tuning
from utils.config import Config

def evaluate(pi_0, pi_e, x, a, r, true_value, config: Config):
    results = []
    progress_bar = tqdm.tqdm(config.estimators, leave=False)
    for estimator in progress_bar:
        estimator_class = getattr(estimators, estimator.name)
        progress_bar.set_description(f"Estimator: {estimator.printname}")
        
        if estimator.hyperparameters == "dynamic":
            estimator.hyperparameters = estimator_class(**estimator.kwargs).get_candidate_hyperparameters(pi_0=pi_0, pi_e=pi_e, x=x, a=a, r=r)
            
        if estimator.tuning is None:    
            for h in ParameterGrid(estimator.hyperparameters):
                start = time.time()
                pi_e_hat = estimator_class(**estimator.kwargs, **h)(pi_0=pi_0, pi_e=pi_e, x=x, a=a, r=r)
                results.append({"Estimator": estimator.printname, "MSE": np.mean(true_value - pi_e_hat)**2, **h, "seconds": time.time() - start})
        else:
            tuner_class = getattr(tuning, estimator.tuning.tuner)
            start = time.time()
            tuner = tuner_class(estimator=estimator_class, estimator_kwargs=estimator.kwargs, hyperparameters=estimator.hyperparameters, **estimator.tuning.kwargs)
            pi_e_hat = tuner(pi_0=pi_0, pi_e=pi_e, x=x, a=a, r=r)
            results.append({"Estimator": estimator.printname, "MSE": np.mean(true_value - pi_e_hat)**2, "seconds": time.time() - start})            
    
    estimators.reward_model.RewardModel.clear_cache()
    results = pd.DataFrame(results)
    results["regret"] = results["MSE"] - results["MSE"].min()
    return results
    