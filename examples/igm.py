#Author Aji Palar

#Our Goal is to map physical interaction networks of proteins and their small molecule ligands
#We expect our method to accuratley predict molecular interaction networks via a Bayesian inferenctial
#approach. We will maximized the accuracy, precition, completness, by using all information availible

from typing import List, Tuple, NewType, Iterable
import numpy as np
from jax import jit
import jax; jax.config.update('jax_platform_name', 'cpu')
import igraph as ig
#Representation

#Number of vertices, proteins, ligands
P: int = 2
L: int = 2
V: int = P + L
#Maximum number of edges
Emax: int = V*(V - 1)//2
print(Emax)

Edge = NewType('Edge', (int, int))

adj_matrix = np.zeros((V, V), dtype=np.int64)

def enumerate_ordered_vertex_pairs(V: int) -> Iterable[Edge]:
    a = np.arange(V, dtype=np.int64) + 1
    ordered_pairs = np.zeros((Emax, 2), dtype=np.int64)
    c = 0
    for i in a:
        for j in range(i + 1, V + 1):
            ordered_pairs[c] = [i, j]
            c+=1
    return ordered_pairs

j_enumerate_ordered_pairs = jit(enumerate_ordered_vertex_pairs)(V)
ordered_pairs = enumerate_ordered_vertex_pairs(V)

g = ig.Graph(V)

