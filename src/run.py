import argparse
import importlib
import os
import time

import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split, ParameterGrid

from datasets.preprocessing import preprocess
from utils.sampler import sample_from_dataset
from utils.policy import get_policy
from utils.eval import evaluate
from utils.store_results import store_results
from utils.config import load_config



if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='Script arguments')
    parser.add_argument("--config", "-c", type=str, default="configs/synthetic.yaml", help="Config file path")
    parser.add_argument("--saved-experiment", "-s", type=str, default=None, help="Saved config or figure path")
    args = parser.parse_args()
    if not args.saved_experiment:
        config = load_config(args.config)
        results = []
        
        for i in tqdm.tqdm(range(config.experiment.n_iter), desc="Iteration"):
            ablation_grid = tqdm.tqdm(ParameterGrid(config.experiment.ablation), leave=False)
            for ablation in ablation_grid:
                # Ablate the config
                ablation_grid.set_description(f"{ablation}")
                for ablation_key, ablation_value in ablation.items():
                    if type(ablation_value) == str:
                        ablation_value = f'"{ablation_value}"' 
                    exec(f"config.{ablation_key} = {ablation_value}")

                # Prepare the dataset
                np.random.seed(i)                
                dataset = importlib.import_module(f"datasets.{config.dataset.module}")
                X, y = dataset.get_dataset(**config.dataset.kwargs)
                for preprocessing_step in config.dataset.preprocessing:
                    X, y = preprocess(X, y, preprocessing_step.name, **preprocessing_step.kwargs)
                
                # Split the data for OPE and policy learning    
                a_num = len(np.unique(y)) if len(y.shape) == 1 else y.shape[1]
                X_eval, X_model, y_eval, y_model = train_test_split(X, y, train_size=config.dataset.eval_model_split, random_state=i)
                
                # Train the policies
                pi_0 = get_policy(config.logging_policy.name, X=X_model, y=y_model, a_num=a_num, **config.logging_policy.kwargs)
                pi_e = get_policy(config.target_policy.name, X=X_model, y=y_model, a_num=a_num, **config.target_policy.kwargs)
                
                # Simulate the logging policy and get the ground truth target policy value
                x, a, r, true_value = sample_from_dataset(X_eval, y_eval, pi_0, pi_e, a_num=a_num, **config.sampler)
                
                # Evaluate the estimators
                result = evaluate(pi_0, pi_e, x, a, r, true_value, config)
                result = result.assign(**ablation)
                results.append(result)
        df = pd.concat(results)
        store_results(df, load_config(args.config)) 
    else:
        base_identifier = args.saved_experiment.replace("results/figures/", "").replace("results/configs/", "").replace("results/data/", "").replace(".pdf", "").replace(".yaml", "").replace(".csv", "")
        config = load_config(os.path.join("results/configs", base_identifier + ".yaml"))
        df = pd.read_csv(os.path.join("results/data", base_identifier + ".csv"))
        store_results(df, config) 
    print(f"Finished in {int(time.time() - start)} seconds")    
