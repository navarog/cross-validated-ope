# Cross-Validated Off-Policy Evaluation

This repository contains the code to reproduce the results reported in [our paper](http://arxiv.org/abs/2405.15332). We study the problem of estimator selection and hyper-parameter tuning in off-policy evaluation. Although cross-validation is the most popular method for model selection in supervised learning, off-policy evaluation relies mostly on theory-based approaches, which provide only limited guidance to practitioners. We show how to use cross-validation for off-policy evaluation. This challenges a popular belief that cross-validation in off-policy evaluation is not feasible. We evaluate our method empirically and show that it addresses a variety of use cases.

## Installation and Running the Experiments

The code is written in python and uses standard ML libraries + PyTorch. We recommend creating a new python environment to ensure the latest versions of all libraries. To install all dependencies, run:
```bash
pip install -r requirements.txt
```
The code is tested with python version 3.10+.
The simplest example to run is:
```bash
python src/run.py --config configs/example.yaml
```
It should finish within a few minutes. After that, the figure is stored in the `results/figures/example` directory. The data to produce this figure is stored under the `results/data/example` directory and the used config under `results/configs/example`.
In all cases, the file identifier of the run is the timestamp at the time of producing the figure.

### Replotting the Results

*Note: the experiments assume you have LaTeX installed with a proper path provided to print math labels in the plots. You can disable this by changing the config `plot.use_latex` value to `False`.*

You can use the saved data files to skip the computations and plot the experiment results from existing data, for example, if you want to change the variable that is being aggregated over. Run the following command to reproduce the results in Figure 3 without extensive computations:

```bash
python src/run.py --saved-experiment results/figures/tuning/2024-05-19T16-06-35.122797.pdf
```

The figure should appear in the `results/figures/tuning` directory. The `--saved-experiment` argument is the path to either config, data, or the figure file you wish to reproduce. The command automatically loads the corresponding config and the data, replots the figure, and saves everything under the current timestamp.
For example, if you wish to change the figure title, you can change it in the saved config file and re-run the plotting with that config.

## To Produce the Figures from the Paper
Figure 1:
```bash
python src/run.py --config configs/dr_strong.yaml
```

Figure 2:
```bash
python src/run.py --config configs/dr_weak.yaml
```

Figure 3:
```bash
python src/run.py --config configs/tuning.yaml
```

Figure 4:
```bash
python src/run.py --config configs/ablation
```

Figure 5:
```bash
python src/run.py --config configs/k_splits.yaml
```

Figure 6:
```bash
python src/run.py --config configs/ocv_dm.yaml
```
You will find the results from these runs already precomputed under the `results` folder. 

## Experiment Configuration Explained

Feel free to experiment with the configurations provided. Change a few variables or create your own. We describe an [example config](configs/dr_strong.yaml) that reproduces Figure 1 in detail below. 
```yaml
### The whole config is loaded into src/utils/config.py Config dataclass.
experiment:
  directory: dr_strong # Identifier, used for storing the results.
  n_iter: 500 # Number of repeated independent runs.
  ablation: # Ablation object that is loaded by `sklearn.model_selection.ParameterGrid`. Check it for a detailed explanation of possible types.
    dataset.kwargs["dataset"]: [ecoli, glass, letter, optdigits, page-blocks, pendigits, satimage, vehicle, yeast] # This will change the specified key in the config for every value in the list. You can specify multiple keys at once, see `configs/tuning.yaml`.
dataset:
  module: openml # Name of the file under src/datasets to load the data loading function.
  kwargs: # Hyper-parameters passed to the data loading function.
    dataset: ecoli # This value is completely ignored as the ablation changes it.
    version: 1
  preprocessing: # Preprocessing functions called on the dataset before it is passed to policies.
    - name: ordinal_label # Preprocessing function identifier from datasets/preprocessing.py.
  eval_model_split: 0.5 # Split ratio where `eval` is the set for evaluation and `model` is the set for policy learning.
logging_policy:
  name: softmax # Policy identifier from src/utils/policy.py get_policy function.
  kwargs: # Policy function arguments
    temperature: 1
    model_kwargs:
      max_iter: 1000000
    preprocessing: # Preprocessing functions called specifically before this policy learning.
      - name: subsample
        kwargs:
          n: 1.0
          replace: True
target_policy:
  name: softmax
  kwargs:
    temperature: 10
    model_kwargs:
      max_iter: 1000000
    preprocessing:
      - name: subsample
        kwargs:
          n: 1.0
          replace: True
estimators: # List of estimators to evaluate.
  - name: IPS # The name identifies the class to load as defined in src/estimators/__init__.py file.
  - name: DM
  - name: DR
  - name: EstimatorSelection
    printname: \ensuremath{\tt OCV_{IPS}} # The name this estimator has in legend when plotted.
    hyperparameters: # Hyper-parameter grid for this estimator loaded by `sklearn.model_selection.ParameterGrid`. Can have multiple keys specified. Each combination is then passed to the method as its **kwargs.
      - estimator: [IPS, DM, DR]
    tuning: # If no tuning method is provided, the estimator is evaluated for all hyper-parameter combinations from its grid and all results are logged.
      tuner: OCV # Tuning method that gets imported from the src/tuning/__init__.py file.
      kwargs: # Tuning method hyper-parameters.
        valid_estimator: IPS
  - name: EstimatorSelection
    printname: \ensuremath{\tt OCV_{DR}}
    hyperparameters:
      - estimator: [IPS, DM, DR]
    tuning:
      tuner: OCV
      kwargs:
        valid_estimator: DR
  - name: EstimatorSelection
    printname: \textsc{Slope}
    hyperparameters:
      - estimator: [IPS, DR, DM]
    tuning:
      tuner: SLOPE
  - name: EstimatorSelection
    printname: \ensuremath{\tt PAS{\text -}IF}
    hyperparameters:
      - estimator: [IPS, DM, DR]
    tuning:
      tuner: PASIF
plot:
  type: catplot # Type of plot to use, identifiers are defined in src/utils/store_results.py file. 
  xlabel: Dataset
  use_latex: False # Whether to use LaTeX in matplotlib for printed texts. See https://matplotlib.org/stable/users/explain/text/usetex.html for more details.

### The complete list of configurable parameters with their default values can be found in the src/utils/config.py file.  
```

## Developing a New Tuning Method/Estimator
It is straightforward to implement a new method into this codebase. The easiest way is to copy the existing estimator, for example, SwitchDR, and change the bodies of its methods. Then, do not forget to import the class in the [\_\_init__.py file](src/estimators/__init__.py). The same follows for a new tuning method.

## BibTeX
```
@misc{cief_cross-validated_2024,
	title = {Cross-Validated Off-Policy Evaluation},
	url = {http://arxiv.org/abs/2405.15332},
	number = {{arXiv}:2405.15332},
	publisher = {{arXiv}},
	author = {Cief, Matej and Kompan, Michal and Kveton, Branislav},
	urldate = {2024-05-27},
	date = {2024-05-24},
	eprinttype = {arxiv},
	eprint = {2405.15332 [cs]},
	keywords = {Computer Science - Machine Learning},
}
```