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



class TestAIS(IMP.test.TestCase):
    """Test the various functions in the AIS module""" 
    
    def test_do_ais(self):
        mu = 5
        sigma = 2
        fn_pdf = jax.scipy.stats.norm.pdf
        n_samples = 100
        n_inter = 50
        betas = np.linspace(0, 1, n_inter)
        key = jax.random.PRNGKey(10)
        T = ais.T



        samples, weights = ais.do_ais(key = key, 
                                      mu = mu,
                                      sigma = sigma,
                                      n_samples = n_samples, 
                                      n_inter = n_inter, 
                                      betas = betas,
                                      f0_pdf = ais.f0_pdf,
                                      fj_pdf = ais.fj_pdf,
                                      fn_pdf = ais.fn_pdf,
                                      T = T)

        atol_l = [10 ** i for i in range(1, -6, -1)]
        rtol_l = [i * 10 ** -1 for i in range(9, 1, -1)]

        mean = ais.get_mean(samples, weights)
        for rtol in rtol_l:
            np.testing.isclose(mean, mu, rtol = rtol)
            

    def test_get_ais_mean(self):
        pass

        
if __name__ == '__main__':
    IMP.test.main()
