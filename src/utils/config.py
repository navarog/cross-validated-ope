import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Union

from dacite import from_dict

    
@dataclass
class Experiment:
    n_iter: int
    directory: str
    ablation: Union[str, Dict[str, List], List[Dict[str, List]]] = field(default_factory=dict)

@dataclass
class Preprocessing:
    name: str
    kwargs: Dict = field(default_factory=dict)

@dataclass
class Dataset:
    module: str
    kwargs: Dict = field(default_factory=dict)
    preprocessing: List[Preprocessing] = field(default_factory=list)
    eval_model_split: Union[float, int] = 0.5

@dataclass
class Policy:
    name: str
    kwargs: Dict = field(default_factory=dict)
    
@dataclass
class Tuning:
    tuner: str
    kwargs: Dict = field(default_factory=dict)

@dataclass
class Estimator:
    name: str
    printname: str = None
    kwargs: Dict = field(default_factory=dict)
    hyperparameters: Union[str, Dict[str, List], List[Dict[str, List]]] = field(default_factory=dict)
    tuning: Union[Tuning, None] = None
    
    def __post_init__(self):
        if self.printname is None:
            self.printname = self.name

@dataclass
class Plot:
    type: str = "table"
    xlabel: Union[str, None] = None
    ylabel:  Union[str, None] = "MSE"
    xvar:  Union[str, None] = None
    yvar:  Union[str, None] = "MSE"
    xscale:  Union[str, None] = None
    yscale:  Union[str, None] = "log"
    title:  Union[str, None] = None
    figsize: List = field(default_factory=lambda: [4, 8/3])
    use_latex: bool = True

@dataclass
class Config:
    experiment: Experiment
    dataset: Dataset
    logging_policy: Policy
    target_policy: Policy
    estimators: List[Estimator]
    sampler: Dict = field(default_factory=dict)
    plot: Plot = field(default_factory=Plot)
    
def load_config(config_path: str) -> Config:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return from_dict(data_class=Config, data=config)

def save_config(config: Config, config_path: str) -> None:
    with open(config_path, "w") as file:
        yaml.dump(asdict(config), file)
