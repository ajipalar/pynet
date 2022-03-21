from pathlib import Path
import pyext.src.data_in
import numpy as np
from .typedefs import Array

def write_example_benchmark():
    """Write an an example benchmark file
    to definen format
    """

    fpath = benchmark / "ppi_benchmark_example.tsv"
    assert fpath.is_file is False, f"{fpath} exists\nAborting"

    with open(fpath, 'w') as f:
        line = "interaction-id\t"


# per sample test statistics
# TP known
# FN known
# TN unkown
# FP unkown

def accuracy(y_ref: Array, y_pred: Array) -> float:
    """
    @param y_ref 1d array like
    @param y_pred 1d array like
    @return accuracy_score 
    """
    assert y_ref.shape == y_pred.shape
    assert y_ref.ndim == 1
    assert y_pred.ndim == 1
    assert y_ref.dtype == y_pred.dtype

    score: Array  = y_ref ==  y_pred  # element wise comparison 
    accuracy_score: float  = np.sum(score)
    accuracy_score /= len(score)
    return accuracy_score
    
