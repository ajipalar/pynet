from __future__ import print_function
import IMP.test
import IMP.algebra
try:
    import IMP.pynet
    import IMP.pynet.ais as ais
    import IMP.pynet.functional_gibbslib as fg
    import IMP.pynet.PlotBioGridStatsLib as bsl
except ModuleNotFoundError:
    import pyext.src.ais as ais
    import pyext.src.functional_gibbslib as fg
    import pyext.src.PlotBioGridStatsLib as nblib

import io
import jax
import jax.numpy as jnp
import math
import numpy as np
from functools import partial 



class TestAIS(IMP.test.TestCase):
    """Test the various functions in the AIS module""" 

    def test_f0_pdf__j(self):
        mu = 10
        sig = 2

        cases1 = [mu -1, mu, mu + 1]
        cases2 = [mu - 10, mu, mu + 10]

        f0 = ais.f0_pdf__j
        f0 = partial(f0, mu = mu, sig = sig)

        assert f0(cases1[0]) > f0(cases2[0])
        assert f0(cases1[1]) == f0(cases2[1])
        assert f0(cases1[2]) > f0(cases2[2])

        for n in jnp.arange(-1000, 1000):
            assert 0 <= f0(n) <= 1

    def test_fn_pdf__j(self):
        
        mu = 0
        sig = 1

        cases1 = [mu -1, mu, mu + 1]
        cases2 = [mu - 10, mu, mu + 10]

        f0 = ais.fn_pdf__j

        assert f0(cases1[0]) > f0(cases2[0])
        assert f0(cases1[1]) == f0(cases2[1])
        assert f0(cases1[2]) > f0(cases2[2])

        for n in jnp.arange(-1000, 1000):
            assert 0 <= f0(n) <= 1

            
            


    @IMP.test.skip
    def test_T_nsteps__unorm2unorm__p(self):
        mu = 10
        sig = 2

        T_j = ais.T_nsteps_mh__unorm2unorm__p(mu, sig)

        x = jnp.arange(0, 20)
        betas = jnp.arange(0, 1, 0.1)

        maximum = 0.0

        key = jax.random.PRNGKey(1)

        for i, xi in enumerate(x):
            for j, bij in enumerate(betas):
                key, subkey = jax.random.split(key, 2)
                xij = T_j(key, xi, kwargs_intermediate__j = {'beta': bij})

    @IMP.test.skip
    def test_T_nsteps_mh__g(self):
        key = jax.random.PRNGKey(20)
        x = 10
        ij = ais.fj_pdf__g
        ij__j = partial(ij, source__j = ais.fn_pdf__j,
                target__j = ais.f0_pdf__j)

        irvs = jax.random.normal
        kwargs = {}

        test_func = ais.T_nsteps_mh__g
        kwargs_test_func = {'key': key, 'x': x, 'intermediate_rvs__j' : irvs,
                'intermediate__j': ij,  'kwargs_intermediate__j' : kwargs}

        test_func__j = partial(test_func, intermediate__j = ij__j) 

        jresults = jax.jit(test_func)(**kwargs_test_func)
        assert jresults == test_func(**kwargs_test_func)

    
    @IMP.test.skip
    def test_do_ais(self):
        mu = 5
        sigma = 2
        fn_pdf = jax.scipy.stats.norm.pdf
        n_samples = 100
        n_inter = 50
        betas = np.linspace(0, 1, n_inter)
        key = jax.random.PRNGKey(10)
        T = ais.T_nsteps_mh__g

        fj_pdf__j = partial(ais.fj_pdf__g, source__j = jax.scipy.stats.norm.pdf,
                target__j = ais.f0_pdf__j)





        samples, weights = ais.do_ais__g(key = key, 
                                      mu = mu,
                                      sigma = sigma,
                                      n_samples = n_samples, 
                                      n_inter = n_inter, 
                                      betas = betas,
                                      f0_pdf = ais.f0_pdf__j,
                                      fj_pdf = fj_pdf__j,
                                      fn_pdf = ais.fn_pdf,
                                      transition_rule__j = T)

        atol_l = [10 ** i for i in range(1, -6, -1)]
        rtol_l = [i * 10 ** -1 for i in range(9, 1, -1)]

        mean = ais.get_mean(samples, weights)
        for rtol in rtol_l:
            np.testing.isclose(mean, mu, rtol = rtol)
            

    def test_get_ais_mean(self):
        pass

    def test_do_ais_in_normal_context(self):
        ...

        
if __name__ == '__main__':
    IMP.test.main()
