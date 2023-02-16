import numpy as np
import model_proto as mp

n = 10
A = np.zeros((n, n), dtype=np.int8)
A[(1, 2, 3, 4, 5), (0, 0, 0, 1, 2)] = 1
A = np.tril(A, k=-1)
A = A + A.T
Cs = np.array([0, 1, 3], dtype=np.int8)

Ss = np.array([1, 0.7, 0.6, 0.8, 0.5, 0.3, 0.2, 0.1, 0.4, 0.05])
assert len(Ss) == n


