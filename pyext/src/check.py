"""
Perform runtime checks
"""
import numpy as np
import pyext.src.matrix as mat

def cov(m):
    assert np.alltrue(m[np.diag_indices(len(m))] > 0), f"fail : diag"
    assert mat.is_positive_definite(m), f"fail pos"



