import IMP.test

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy as sp

import pyext.src.pynet_rng as pynet_rng
import pyext.src.matrix as mat


def probabilistic_test_chi2(dof):
    print(f"Chi2 probabilistic test {dof}")

    seed = 15
    key = jax.random.PRNGKey(seed)

    arr = pynet_rng.chi2(key, dof, shape=[100000])
    np.testing.assert_almost_equal(np.mean(arr), dof, 1, err_msg=f"mean k={dof}")
    var = np.var(arr)
    try:
        np.testing.assert_almost_equal(var, 2 * dof, 1, err_msg=f"var k={dof}")
    except AssertionError:
        np.testing.assert_almost_equal(var, 2 * dof, 0, err_msg=f"var k={dof}")
        print(f"  Exception: dof {dof} var {var} within 1.5")

    if dof > 1:
        np.testing.assert_almost_equal(
            np.median(arr), dof * (1 - 2 / (9 * dof)) ** 3, 1, err_msg=f"median k={dof}"
        )
    np.testing.assert_almost_equal(
        sp.stats.skew(arr), np.sqrt(8 / dof), 1, err_msg=f"skew k={dof}"
    )
    np.testing.assert_almost_equal(
        sp.stats.kurtosis(arr), 12 / dof, decimal=0, err_msg=f"kurtosis k={dof}"
    )


def probabilistic_test_wishart(key, V, n, p):

    # Assert V is positive Definite
    assert p > 1
    assert n > p - 1
    assert np.alltrue(V == V.T)
    assert mat.is_positive_definite(V)

    S = pynet_rng.wishart(key, V, n, p)
    assert len(S) == p
    assert np.alltrue(S == S.T)
    assert np.alltrue(np.isnan(sp.linalg.cholesky(S)) == False)


class RandomUnitTests(IMP.test.TestCase):
    key = jax.random.PRNGKey(44)

    def test_chi2(self):
        probabilistic_test_chi2(1)
        probabilistic_test_chi2(2)
        probabilistic_test_chi2(3)
        probabilistic_test_chi2(4)
        probabilistic_test_chi2(7)
        probabilistic_test_chi2(11)
        probabilistic_test_chi2(22)
        probabilistic_test_chi2(98)

    def test_wishart_1(self):
        key = self.key

        V = jnp.array([[1, 0], [0, 1]])
        p = len(V)
        n = p
        probabilistic_test_wishart(key, V, n, p)

    def test_wishart_2(self):
        V = jnp.array([[1, 0], [0, 1]])
        probabilistic_test_wishart(self.key, V, 3, 2)

    def test_wishart_3(self):
        V = jnp.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        probabilistic_test_wishart(self.key, V, 3, 3)

    def test_wishart_4(self):
        V = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        probabilistic_test_wishart(self.key, V, 2, 2)
