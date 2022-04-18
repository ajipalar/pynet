from __future__ import print_function
import IMP.test
import IMP.algebra

try:
    import IMP.pynet
except ModuleNotFoundError:
    import pyext.src.functional_gibbslib as fg

    pass

from functools import partial
import io
import jax
import jax.numpy as jnp
import numpy as np
import os
import math


class TestFunctionalGibbsLib(IMP.test.TestCase):
    def test_generic_functional_gibbs(self):
        """tests generic_gibbs, generic_gibbsf and gibbs"""

        init_func = fg.example_generic_init_params
        update_func = fg.example_generic_update_params

        seed = 11
        nsamples = 10
        thin_every = 10
        nparams = 2

        # partial application of samplers

        def apply_partial(f):
            p_f = partial(
                f,
                nsamples=nsamples,
                thin_every=thin_every,
                nparams=nparams,
                init_params=init_func,
                update_params=update_func,
            )
            return p_f

        p_generic_gibbsf = apply_partial(fg.generic_gibbsf)
        j_generic_gibbsf = jax.jit(p_generic_gibbsf)

        p_generic_gibbs = apply_partial(fg.generic_gibbs)

        gibbs = fg.gibbs
        gibbsf = fg.gibbsf

        p_gibbsf = partial(gibbsf, N=10, thin=10, rho=0.3)

        key = jax.random.PRNGKey(seed)
        s_1 = p_generic_gibbsf(key)
        assert jnp.all(key == jax.random.PRNGKey(seed))
        j_1 = j_generic_gibbsf(key)
        assert jnp.all(key == jax.random.PRNGKey(seed))
        s_2 = p_generic_gibbs(key)
        assert jnp.all(key == jax.random.PRNGKey(seed))
        s_3 = gibbs(key)
        assert jnp.all(key == jax.random.PRNGKey(seed))
        # s_4 = p_gibbsf(key)

        print("p_generic_gibbsf")
        print(s_1)
        print("j_generic_gibbsf")
        print(j_1)
        print("p_generic_gibbs")
        print(s_2)
        print("gibbs")
        print(s_3)
        # print('p_gibbsf')
        # print(s_4)

        # Passes to 1e-06 precision

        # tolerance = [1e-10, 1e-09, 1e-08,
        #             1e-07, 1e-06, 1e-05,
        tolerance = [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01]

        tolerance.reverse()

        results = [s_1, j_1, s_2, s_3]  # , s_4]

        for rtol in tolerance:
            print(f"testing {rtol}")
            for i in range(len(results)):
                for j in range(i, len(results)):
                    # print(i, j)
                    assert jnp.allclose(results[i], results[j], rtol=rtol)


if __name__ == "__main__":
    IMP.test.main()
