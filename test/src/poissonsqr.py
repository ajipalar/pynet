from __future__ import print_function
import IMP
import IMP.test
import jax.numpy as jnp
import jax.random
import numpy as np
from hypothesis import given, strategies as st
from typing import Any
from functools import partial
import collections

Module = Any

def remove_ith_entry(ndarray, i: int, src: Module):
    arr_l = len(ndarray)
    assert 0<=i<arr_l
    #test_f = partial(src.remove_ith_entry, arr_l=arr_l)
    test_f = src.remove_ith_entry__s(arr_l)
    out_arr = test_f(arr=ndarray,  i=i)
    out_arr = np.array(out_arr)
    assert len(out_arr) == arr_l -1

    ground_truth1 = ndarray[0:i]
    ground_truth2 = ndarray[i+1:arr_l]
    
    if len(ground_truth1 > 0):
        np.testing.assert_almost_equal(ground_truth1, out_arr[0:i])
    if len(ground_truth2 > 0):
        np.testing.assert_almost_equal(ground_truth2, out_arr[i:arr_l])


def i_eq_0(arr, arr_l, src: Module, kwds):
    """Test defniition for i_eq_0 function in poissonsqr"""
    out_arr = src.i_eq_0(arr, arr_l) 
    assert len(out_arr) == arr_l - 1
    np.testing.assert_almost_equal(arr[1], out_arr[0], decimal=kwds.decimal)
    np.testing.assert_almost_equal(arr[-1], out_arr[-1], decimal=kwds.decimal)
    np.testing.assert_almost_equal(arr[1:arr_l], out_arr, decimal=kwds.decimal)


def i_eq_arr_l(arr, arr_l, src: Module):
    out_arr = src.i_eq_arr_l(arr, arr_l)
    np.testing.assert_almost_equal(arr[0:arr_l -1], out_arr)

def zero_lt_i_lt_arr_l(arr, arr_l, i: int):
    out_arr = src.zero_lt_i_lt_arr_l(arr, arr_l, i)
    assert len(out_arr) == arr_l -1
    assert out_arr[0] == arr[0]
    assert out_arr[-1] == arr[-1]

def branch2(arr, arr_l, i, src: Module):
    out_arr = src.branch2(arr, arr_l, i)
    eq_arr = src.i_eq_arr_l


def i_eq_0__s(arr, arr_l, src: Module, kwds):
    assert arr_l > 1
    i_eq_0__j = src.i_eq_0__s(arr_l)
    ji_eq_0 = jax.jit(i_eq_0__j)
    jarr = ji_eq_0(arr).block_until_ready()

    jarr = np.array(jarr)
    py_arr = src.i_eq_0(arr, arr_l)
    py_arr = np.array(py_arr)
    arr = np.array(arr)
    
    np.testing.assert_almost_equal(jarr, py_arr, decimal=kwds.decimal)
    np.testing.assert_almost_equal(py_arr, jarr, decimal=kwds.decimal)
    np.testing.assert_almost_equal(jarr, arr[1:arr_l], decimal=kwds.decimal)
    np.testing.assert_almost_equal(arr[1:arr_l], jarr, decimal=kwds.decimal)




class PoissUnitTests(IMP.test.TestCase):
    """Base class for Poisson SQR unit tests"""
    src = None  # Derived class overides this
    rtol = None
    atol = None
    decimal = None
    kwds = None

    def test_i_eq_0__s(self):
        # case 1
        arr = np.random.rand(8)
        arr_l = len(arr)
        i_eq_0__s(arr, arr_l, self.src, self.kwds)

        # case 2
        arr = np.random.rand(3)
        arr_l = len(arr)
        i_eq_0__s(arr, arr_l, self.src, self.kwds)

        # case 3
        key = jax.random.PRNGKey(10)
        darr = jax.random.normal(key, shape=(8,)).block_until_ready()
        i_eq_0__s(darr, len(darr), self.src, self.kwds)

        # case 4
        arr = np.random.rand((9*9)).reshape((9, 9))
        arr_l = 9
        i_eq_0__s(arr, arr_l, self.src, self.kwds)

        # case 5
        darr = jax.random.normal(key, shape=(4, 4))
        i_eq_0__s(darr, len(darr), self.src, self.kwds)



    @IMP.test.skip
    def test_i_eq_0(self):
        # case 1
        arr = np.random.rand(8)
        arr_l = len(arr)
        i_eq_0(arr, arr_l, self.src)

        # case 2
        arr = np.random.rand(3)
        arr_l = len(arr)
        i_eq_0(arr, arr_l, self.src)

        # case 3
        key = jax.random.PRNGKey(10)
        darr = jax.random.normal(key, shape=(8,)).block_until_ready()

        arr_l = len(darr)
        #i_eq_0(darr, arr_l, self.src)

    @IMP.test.skip
    def test_i_eq_arr_l(self):
        # case 1
        arr = np.random.rand(8)
        arr_l = len(arr)
        i_eq_arr_l(arr, arr_l, self.src)

        # case 2
        arr = np.random.rand(3)
        arr_l = len(arr)
        i_eq_arr_l(arr, arr_l, self.src)

        # case 3
        key = jax.random.PRNGKey(10)
        darr = jax.random.normal(key, shape=(8,)).block_until_ready()

        arr_l = len(darr)
        i_eq_arr_l(darr, arr_l, self.src)

        

    def test_remove_ith_entry(self):
        n = np.random.rand(24)
        for i in range(0,24):
            remove_ith_entry(n, i, self.src)

    @IMP.test.skip
    def test_remove_ith_entry1(self):
        n = np.random.rand(4)
        remove_ith_entry(n, 1, self.src)


    @IMP.test.skip
    def test_remove_ith_entry_matrix(self):
        n = np.random.rand((7*7)).reshape((7, 7))
        for i in range(1, 7):
            remove_ith_entry(n, i, self.src)




class PoissPropTests(IMP.test.TestCase):
    """Base class for Poisson SQR Property tests"""



