import IMP.test

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy as sp

import pyext.src.random

def probablistic_test_chi2():

    seed = 15
    key = jax.random.PRNGKey(seed)

    arr = pyext.src.random.chi2(key, 4, shape=[100000])
    np.testing.assert_almost_equal(np.mean(arr), 4, 1)

class RandomUnitTests(IMP.test.TestCase):
    def test_chi2(self):
        probablistic_test_chi2()
    

