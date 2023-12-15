import numpy as np
from libsvmdata import fetch_libsvm
from sklearn.datasets import fetch_california_housing
import appdirs
import numpy as np
from download import download
from rpy2 import robjects
from rpy2.robjects import numpy2ri
from scipy.sparse import csc_array
import os

def get_data(dataset):
    print(f'Preparing {dataset} dataset...')
    if dataset == 'leukemia': 
        X, y = fetch_libsvm('leukemia')
    elif dataset == 'breheny1':
        X, y = fetch_breheny('Scheetz2006')
    elif dataset == 'breheny2':
        X, y = fetch_breheny('Rhee2006')
        X = X.A
    elif dataset == 'housing':
        data = fetch_california_housing()
        X, y = data.data, data.target
    print('Done.')
    print(f'X is composed of {X.shape[0]} samples and {X.shape[1]} features.')
    print(f'The format of X is {type(X)}')
    print(f'y is of dtype: {y.dtype}, max is {y.max()}, min is {y.min()}')
    return X, y



def fetch_breheny(dataset: str):
    # this code is taken from https://github.com/benchopt/benchmark_lasso_path/blob/main/datasets/breheny.py
    base_dir = appdirs.user_cache_dir("benchmark_lasso_path")

    path = os.path.join(base_dir, dataset + ".rds")

    # download raw data unless it is stored in data folder already
    if not os.path.isfile(path):
        url = "https://s3.amazonaws.com/pbreheny-data-sets/" + dataset + ".rds"
        download(url, path)

    read_rds = robjects.r["readRDS"]
    numpy2ri.activate()

    data = read_rds(path)
    X = data[0]
    y = data[1]

    density = np.sum(X != 0) / X.size

    if density <= 0.2:
        X = csc_array(X)

    return X, y
