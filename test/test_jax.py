from __future__ import print_function
import IMP
import IMP.test
import IMP.algebra
import os
import math

from functools import partial
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax
from jax import grad, jit, pmap, vmap
from jax.experimental.maps import xmap, mesh
from jax.experimental import sparse
import unittest

from typing import Any, Callable

import numpy as np
class ArrayType:
    def __getitem__(self, idx):
        return idx

f32 = ArrayType()
i32 = ArrayType()

class TestXMAP(IMP.test.TestCase):
    """xmap is an experimental feature that enables named axis for numpy arrays
       dtype[shape] -> dtype[named_shape]
       f32[(1000, 2)] -> f32[{'n_samples': 1000, 'n_params': 2}]
       f32[(1000, 1, 1)] -> f32{'n_samples': 1000, 'x': 1, 'y': 1}]"""

    @unittest.skip
    def test_unary_functions(self):
        
        def uf(x: f32[(0,)] , y: f32[(0,)]) -> f32[(0,)]:
            print(x.shape)
            assert x.shape == (1,)
            assert y.shape == (1,)
            return x + 2 * y

        xdim=200
        ydim=1

        x = jnp.arange(200).reshape((xdim, ydim))
        y = x

        # test uf before xmap

        #uf(x[0, 0], y[0, 0])

        fx = xmap(uf,
                 in_axes=(['a', ...], ['a', ...]),
                 out_axes=['a', ...])

        fxr = fx(x, y)

        def rtest(uf, x, y, fxr):
    
            for i in range(xdim):
                    print(x[i], y[i])

                    assert np.testing.assert_allclose(uf(x[i], y[i]), fxr[i])

        rtest(uf, x, y, fxr)
    
        jfx = jax.jit(fx)
        jxr = jfx(x, y)

        np.testing.assert_allclose(jxr, fxr)

        jgfx = jax.jit(grad(fx))
        guf = grad(uf)
        jfgx = jax.jit(xmap(guf,
                          in_axes = (['a', ...], ['a', ...]),
                                     out_axes=['a', ...]))

        np.testing.assert_allclose(jgfx(x, y), jfgx(x, y))


class TestSparse(IMP.test.TestCase):
    atol=1e-5
    rtol=1e-5

    # BCOO batched coordinate sparse array

    def test_matmul(self):

        M = jnp.array([[0., 1., 0., 2.],
                       [3., 0., 0., 0.],
                       [0., 0., 4., 0.]])
    
        M_sp = sparse.BCOO.fromdense(M)
    
        M_td = M_sp.todense()
    
        np.testing.assert_allclose(M, M_td, atol=self.atol)
        np.testing.assert_allclose((M @ M.T), (M_sp @ M_sp.T).todense(), atol=self.atol)

    def test_add(self):
        """Test element wise addition of sparse matrices"""
        M = jnp.array([[1., 2., 3.],
                       [4., 5., 6.],
                       [0., -1., -4.]])

        M_sp = sparse.BCOO.fromdense(M)

        np.testing.assert_allclose(M + M, (M_sp + M_sp).todense(), atol=self.atol)

    def test_subtract(self):
        """Test element wise subtraction of sparse matrices"""
        M = jnp.array([[1., 2., 3.],
                       [4., 5., 6.],
                       [0., -1., -4.]])

        M_sp = sparse.BCOO.fromdense(M)

        np.testing.assert_allclose(M - M, (M_sp - M_sp).todense(), atol=self.atol)

    def test_multiply(self):
        """Test element wise multiplication of sparse matrices"""
        M = jnp.array([[1., 2., 3.],
                       [4., 5., 6.],
                       [0., -1., -4.]])

        M_sp = sparse.BCOO.fromdense(M)

        np.testing.assert_allclose(M * M, (M_sp * M_sp).todense(), atol=self.atol)

    def test_log(self):
        """Test numpy as jax element wise logarithm of sparse matrices"""
        M = jnp.array([[1., 2., 3.],
                       [4., 5., 6.],
                       [1., 1., 4.]])

        M_sp = sparse.BCOO.fromdense(M)

        np.testing.assert_allclose(jnp.log(M), jnp.log(M_sp.todense()), atol=self.atol)

    def test_sparse_jit(self):
        """Make sure jit compilation works for sparse matrices"""
        M = jnp.array([[1., 2., 3.],
                       [4., 5., 6.],
                       [1., 1., 4.]])

        M_sp = sparse.BCOO.fromdense(M)

        def matmul(a, b):
            return a @ b

        jmatmul = jax.jit(matmul)

        #jit compile jmatmul for jnp.array

        j_mm = jmatmul(M, M).block_until_ready()

        #jit compile jmatmul for sparse matrix

        j_mspmspt = jmatmul(M_sp, M_sp.T).block_until_ready()

        #jit compile jmatmul for mixed types

        j_mspm = jmatmul(M_sp, M).block_until_ready()

        #jit compile jmatmul for other mixed types

        j_mmsp = jmatmul(M, M_sp).block_until_ready()

        np.testing.assert_allclose(j_mm, matmul(M, M), atol=self.atol)
        np.testing.assert_allclose(j_mspmspt.todense(), M @ M.T, atol=self.atol)
        
        



    
        

        # composable with grad

class TestPMAP(IMP.test.TestCase):
    pass


        

if __name__ == '__main__':
    IMP.test.main()
