import IMP.test

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy as sp

import pyext.src.pynet_rng


def probablistic_test_chi2(dof):
    print(f"Chi2 probabilistic test {dof}")

    seed = 15
    key = jax.random.PRNGKey(seed)

    arr = pyext.src.pynet_rng.chi2(key, dof, shape=[100000])
    np.testing.assert_almost_equal(np.mean(arr), dof, 1, err_msg=f"mean k={dof}")
    var = np.var(arr)
    try:
        np.testing.assert_almost_equal(var, 2*dof, 1, err_msg=f"var k={dof}")
    except AssertionError:
        np.testing.assert_almost_equal(var, 2*dof, 0, err_msg=f"var k={dof}")
        print(f"  Exception: dof {dof} var {var} within 1.5")

    if dof > 1:
        np.testing.assert_almost_equal(np.median(arr), dof*(1 - 2/(9 * dof))**3, 1, err_msg=f"median k={dof}")
    np.testing.assert_almost_equal(sp.stats.skew(arr), np.sqrt(8 / dof), 1, err_msg=f"skew k={dof}")
    np.testing.assert_almost_equal(sp.stats.kurtosis(arr), 12 / dof, decimal=0, err_msg=f"kurtosis k={dof}")
    
    

    

class RandomUnitTests(IMP.test.TestCase):
    def test_chi2(self):
        probablistic_test_chi2(1)
        probablistic_test_chi2(2)
        probablistic_test_chi2(3)
        probablistic_test_chi2(4)
        probablistic_test_chi2(7)
        probablistic_test_chi2(11)
        probablistic_test_chi2(22)
        probablistic_test_chi2(98)
    

