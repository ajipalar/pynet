from __future__ import print_function
import IMP
import IMP.test
import numpy as np
import jax.numpy as jnp
from typing import Any

Module = Any

def test_is_potentially_connected(src: Module):

    p = 10
    A = np.zeros((p, p), dtype=int)

    assert not src.is_potentially_connected(A, p)
    A = jnp.zeros((p, p), dtype=int)
    assert not src.is_potentially_connected(A, p)
    A = jnp.ones((p, p), dtype=int)
    assert src.is_potentially_connected(A, p)
    A = np.ones((p, p), dtype=int)

def test_adjacent_nodes(src: Module):
    p = 10
    A = np.zeros((p, p), dtype=int)

    arr, deg = src.adjacent_nodes(0, A, p)
    assert arr.shape == (p,)
    assert deg == 0
    for i in arr:
        assert i == p

    A = jnp.ones((p, p), dtype=int)
    arr, deg = src.adjacent_nodes(0, A, p)
    assert arr.shape == (p,)
    assert deg == p - 1
    for i, j in enumerate(arr):
        assert arr[i] == i + 1



class GraphTests(IMP.test.TestCase):
    def test_potentially_connected(self):
        test_is_potentially_connected(self.src)

    def test_adjacent_nodes(self):
        test_adjacent_nodes(self.src)
