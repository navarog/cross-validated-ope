import numpy as np
from sklearn.datasets import fetch_openml
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def get_dataset(dataset="yeast", version=1):
    X, y = fetch_openml(dataset, version=version, return_X_y=True, as_frame=False, parser='pandas')
    return X, y
