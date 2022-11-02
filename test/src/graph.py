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

class GraphTests(IMP.test.TestCase):
    def test_potentially_connected(self):
        test_is_potentially_connected(self.src)
