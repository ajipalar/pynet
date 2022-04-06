from __future__ import print_function
import IMP
import IMP.test
import jax.numpy as jnp
import jax.random
from jax import jit
import numpy as np
from hypothesis import given, strategies as st
from typing import Any
from functools import partial
import collections

Module = Any

def dev_remove_ith_entry__s(a, src: Module):
    f__j = src.remove_ith_entry__s(a)
    n=len(a)
    assert (a.ndim == 1) or (a.ndim == 2)

    # jit once
    jitf = jit(f__j)
    jitf(a, 0)

    for i in range(len(a)):
        s1  = slice(1, i)
        s2 = slice(0, i) 
        s3 = slice(i+1, n)
        s4 = slice(i, n-1)
        s5 = slice(0, n-1)
        s6 = slice(0, n)
        if a.ndim == 2:
            s1 = (slice(0, n), s1)
            s2 = (slice(0, n), s2)
            s3 = (slice(0, n), s3)
            s4 = (slice(0, n), s4)
            s5 = (slice(0, n), s5)
            s6 = (slice(0, n), s6)

        out = f__j(a, i)
        jout = jitf(a, i)
        out = np.array(out)
        jout = np.array(jout)

        np.testing.assert_almost_equal(out, jout)
        if i==0:
            # np.testing.assert_almost_equal(a[1:i], out[0:i])
            np.testing.assert_almost_equal(a[s1], out[s2])
        if 0<i<=n:
            #np.testing.assert_almost_equal(a[0:i], out[0:i])
            np.testing.assert_almost_equal(a[s2], out[s2])
            #np.testing.assert_almost_equal(a[i+1:n], out[i:n-1])
            np.testing.assert_almost_equal(a[s3], out[s4])
        if i==n:
            #np.testing.assert_almost_equal(a[0:n-1], out[:])
            np.testing.assert_almost_equal(a[s5], out[s6])
            
class PoissUnitTests(IMP.test.TestCase):
    """Base class for Poisson SQR unit tests"""
    src = None  # Derived class overides this
    rtol = None
    atol = None
    decimal = None
    kwds = None
    
    def test_dev_remove_ith_entry(self):
        def run(key, shape):
            key, subkey = jax.random.split(key)
            m = jax.random.normal(key, shape=shape)
            dev_remove_ith_entry__s(m, self.src)
            return key

        key = jax.random.PRNGKey(5)
        key = run(key, (2,))
        key = run(key, (3,))
        key = run(key, (5,))
        key = run(key, (6,))
        key = run(key, (2, 2))
        key = run(key, (3, 3))
        key = run(key, (4, 4))
        #key = run(key, (111, 111))

class IsMatrixCompatible(IMP.test.TestCase):
    """Tests if functions are matrix compatible"""

class PoissPropTests(IMP.test.TestCase):
    """Base class for Poisson SQR Property tests"""
