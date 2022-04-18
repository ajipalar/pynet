from __future__ import print_function

try:
    pass
except ModuleNotFoundError:
    pass

import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
import numpy as np
import unittest

import IMP.test
import IMP.algebra

import io
import os
import math


class TestPoisson(IMP.test.TestCase):
    def test_p100(self):
        lam = 100
        p100 = dist.Poisson(lam)
        n_samples = 5000
        key = jax.random.PRNGKey(10)
        a_list = jnp.zeros(n_samples)
        b_list = jnp.zeros(n_samples)

        print(f"Testing Poisson({lam})...\n")

        for i in range(n_samples):
            key, subkey = jax.random.split(key)
            a = p100.sample(key)
            b = jax.random.poisson(key, 100)
            self.assertEqual(a, b)

            a_list = a_list.at[i].set(a)
            b_list = b_list.at[i].set(b)

            result = jnp.all(a_list == b_list)
            self.assertEqual(result, True)

        epsilon = 5
        av = jnp.mean(a_list)
        var = jnp.var(a_list)
        print(f"mean {av}\nvar{var}")

        self.assertAlmostEqual(av, var, delta=5)


if __name__ == "__main__":
    IMP.test.main()
