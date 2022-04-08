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
    n = len(a)
    assert (a.ndim == 1) or (a.ndim == 2)

    # jit once
    jitf = jit(f__j)
    jitf(a, 0)

    for i in range(len(a)):
        s1 = slice(1, i)
        s2 = slice(0, i)
        s3 = slice(i + 1, n)
        s4 = slice(i, n - 1)
        s5 = slice(0, n - 1)
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
        if i == 0:
            # np.testing.assert_almost_equal(a[1:i], out[0:i])
            np.testing.assert_almost_equal(a[s1], out[s2])
        if 0 < i <= n:
            # np.testing.assert_almost_equal(a[0:i], out[0:i])
            np.testing.assert_almost_equal(a[s2], out[s2])
            # np.testing.assert_almost_equal(a[i+1:n], out[i:n-1])
            np.testing.assert_almost_equal(a[s3], out[s4])
        if i == n:
            # np.testing.assert_almost_equal(a[0:n-1], out[:])
            np.testing.assert_almost_equal(a[s5], out[s6])


def remove_ith_entry__s_vs_value(src: Module, d1=100, d2=10):
    """Tests the removal of the ith entry from a vector and a matrix
    where answers are known"""
    test_dtype=jnp.float32

    a1d = jnp.arange(d1, dtype=test_dtype)
    a2d = jnp.arange(d2 * d2, dtype=test_dtype).reshape((d2, d2))

    f1d__j = src.remove_ith_entry__s(a1d)
    f2d__j = src.remove_ith_entry__s(a2d)

    j1d = jit(f1d__j)
    j2d = jit(f2d__j)

    # jit compile once
    j1d(a1d, 0)
    j2d(a2d, 0)

    del f1d__j
    del f2d__j

    for i in range(d1):
        a1d_min_i = np.array(j1d(a1d, i))
        assert a1d_min_i.shape == (d1 - 1,)
        if i == 0:
            assert a1d[0] == 0
            assert a1d_min_i[0] == 1
            
            ref = a1d[1:d1]
            pred = a1d_min_i
            np.testing.assert_almost_equal(ref, pred)
        elif 0 < i < d1 - 1:
            ref1 = a1d[0:i]
            ref2 = a1d[i + 1 : d1]
            pred1 = a1d_min_i[0:i]
            pred2 = a1d_min_i[i : d1 - 1]
            np.testing.assert_array_almost_equal(ref1, pred1)
            np.testing.assert_array_almost_equal(ref2, pred2)
        # i==d1
        else:
            ref1 = a1d[0 : d1 - 1]
            pred1 = a1d_min_i
            np.testing.assert_almost_equal(ref1, pred1)

    del d1
    del j1d
    del a1d
    del a1d_min_i
    del ref1
    del ref2
    del pred
    del pred1
    del pred2

    for i in range(d2):
        a2d_min_i = np.array(j2d(a2d, i))
        assert a2d_min_i.shape == (d2 - 1, d2 - 1)
        if i == 0:
            ref = a2d[:, 1:d2]
            pred = a2d_min_i
            np.testing.assert_almost_equal(pred, ref)

        elif 0 < i < d2 - 1:
            ref1 = a2d[:, 0:i]
            ref2 = a2d[:, i + 1 : d2]
            pred1 = a2d_min_i[:, 0:i]
            pred2 = a2d_min_i[:, i : d2 - 1]
            np.testing.assert_array_almost_equal(ref1, pred1)
            np.testing.assert_array_almost_equal(ref2, pred2)
        # i==d2
        else:
            ref1 = a2d[0 : d2 - 1]
            pred1 = a1d_min_i
            np.testing.assert_almost_equal(ref1, pred1)


def get_ulog_score__j_is_jittable(theta, phi, x, src):
    ulog_score__j = src.get_ulog_score__s(theta, phi, x)
    jf = jit(ulog_score__j)
    jf(theta, phi, x, 0)


def helper_init_state(key, n, d):
    k1, k2, k3 = jax.random.split(key, 3)

    X = jax.random.poisson(k1, 11, [d, n])
    theta = jax.random.normal(k2, shape=[d])
    phi = jax.random.normal(k3, shape=[d, d])
    return X, theta, phi


class PoissUnitTests(IMP.test.TestCase):
    """Base class for Poisson SQR unit tests"""

    src = None  # Derived class overides this
    rtol = None
    atol = None
    decimal = None
    kwds = None
    key = jax.random.PRNGKey(10)

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
        # key = run(key, (111, 111))

    def test_dev_remove_ith_entry_vs_value(self):
        remove_ith_entry__s_vs_value(src=self.src)
    def test_get_ulog_score_is_jittable(self):
        d = 3
        n = 1
        X, theta, phi = helper_init_state(self.key, n, d)
        assert theta.shape == (d,)
        assert phi.shape == (d, d)
        assert X.shape == (d, n)
        x = X[:, 0]
        assert x.shape == (d,)
        # assert theta.dtype == np.float32
        # assert phi.dtype == np.float32
        # assert x.dtype == np.float32
        get_ulog_score__j_is_jittable(theta, phi, x, self.src)

    def test_logfactorial(self):

        factorial = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]
        lfactorial = np.log(factorial)
        for i, lf in enumerate(lfactorial):
            logfaci = self.src.logfactorial(i)
            np.testing.assert_almost_equal(lf, logfaci, decimal=self.decimal)

    def test_get_eta2__j(self):
        precision = 5
        n = 4
        d = 4
        X, theta, phi = helper_init_state(self.key, n, d)

        x = X[:, 1]
        get_eta2__j = self.src.get_eta2__s(theta, phi, x)
        jf = jax.jit(get_eta2__j)
        jf(theta, phi, x, 0).block_until_ready()

        for i in range(len(theta)):
            a = get_eta2__j(theta, phi, x, i)
            np.testing.assert_almost_equal(a, b, decimal=precision)

    def test_get_eta2__j_values(self):
        d = 3
        xscale = 4.0
        thetascale = 1.0
        phiscale = 1.0
        x = jnp.ones(d, dtype=jnp.int32) * xscale
        theta = jnp.ones(d) * thetascale
        phi = jnp.ones((d, d)) * phiscale

        get_eta2__j = self.src.get_eta2__s(theta, phi, x)

        a = get_eta2__j(theta, phi, x, 0)
        t1 = thetascale
        t2 = 2 * (6.0)
        eta2 = t1 + t2
        np.testing.assert_almost_equal(eta2, a)

    def test_get_ulog_score__j_values(self):
        d = 3
        xscale = 4.0
        thetascale = 1.0
        phiscale = 1.0
        x = jnp.ones(d, dtype=jnp.int32) * xscale
        theta = jnp.ones(d) * thetascale
        phi = jnp.ones((d, d)) * phiscale

        # i = 0
        # phi[i, i] * x[i]
        t1 = phiscale * xscale
        # theta[i] + 2 * rm_i(phi[:, i], i) @ jnp.sqrt(rm_i(x, i))
        t2 = (thetascale + 2 * 6.0) * jnp.sqrt(xscale)
        t3 = jnp.log(4 * 3 * 2 * 1)

        a = t1 + t2 - t3
        get_ulog_score__j = self.src.get_ulog_score__s(theta, phi, x)
        b = get_ulog_score__j(theta, phi, x, 0)
        np.testing.assert_almost_equal(a, b)


class IsMatrixCompatible(IMP.test.TestCase):
    """Tests if functions are matrix compatible"""


class PoissPropTests(IMP.test.TestCase):
    """Base class for Poisson SQR Property tests"""
