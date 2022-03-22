from __future__ import print_function
import IMP.test
import IMP.algebra
try:
    import IMP.pynet
    import IMP.pynet.ais as ais
    import IMP.pynet.functional_gibbslib as fg
    import IMP.pynet.PlotBioGridStatsLib as bsl
    import IMP.pynet.distributions as dist
except ModuleNotFoundError:
    import pyext.src.ais as ais
    import pyext.src.functional_gibbslib as fg
    import pyext.src.PlotBioGridStatsLib as nblib
    import pyext.src.distributions as dist

import io
import jax
import jax.numpy as jnp
import math
import numpy as np
from functools import partial 
from hypothesis import given, settings, strategies as st
from .src import ais_testdefs as td


deadline_ms = 2000

class MediumTestAIS(IMP.test.TestCase):
    """Test the various functions in the AIS module""" 

    rtol = 1e-05
    atol = 1e-05
    trivial_n_samples = 50
    trvial_n_inter = 10

    def test_sample_trivial(self):
        td.sample_trivial(n_samples=self.trivial_n_samples, 
                n_inter=self.trvial_n_inter,
                decimal_tolerance=5)
    
    def test_sample_trivial2(self):
        td.sample_trivial(n_samples=self.trivial_n_samples, 
                n_inter=self.trvial_n_inter,
                decimal_tolerance=6)

    def test_sample_trivial3(self):
        td.sample_trivial(n_samples=self.trivial_n_samples, 
                n_inter=self.trvial_n_inter,
                decimal_tolerance=7)

    def test_sample_trivial4(self):
        td.sample_trivial(n_samples=self.trivial_n_samples, 
                n_inter=self.trvial_n_inter,
                decimal_tolerance=8)

    def test_sample_trivial5(self):
        td.sample_trivial(n_samples=self.trivial_n_samples, 
                n_inter=self.trvial_n_inter,
                decimal_tolerance=9)

    def test_sample_trivial6(self):
        td.sample_trivial(50000, 1, decimal_tolerance=7)

    def test_sample_trivial7(self):
        td.sample_trivial(1, 50000, decimal_tolerance=7)

    def test_sample_trivial7(self):
        td.sample_trivial(225, 225, decimal_tolerance=9)

    def test_not_ones_trivial(self):
        td.not_ones_trivial(n_samples=5, n_inter=5)


    @IMP.test.skip
    @settings(deadline=deadline_ms)
    @given(st.floats(), st.floats(min_value=1e-5))
    def test_nsteps_mh__g(self, mu, sigma):
        n_steps = 100

        key = jax.random.PRNGKey(7)
        x = 0.0

        kwargs_nsteps_mh = {
            'log_intermediate__j': log_intermediate__j,
            'intermediate_rv__j': dist.norm.rv,
            'n_steps': n_steps,
            'kwargs_log_intermediate__j': {}
        }
                

        #Callable
        x = ais.nsteps_mh__g(key, x, **kwargs_nsteps_mh)

        #Jittable
        nsteps_mh__j = partial(ais.nsteps_mh__g, **kwargs_nsteps_mh)
        nsteps_mh = jax.jit(nsteps_mh__j)

        xj = nsteps_mh(key, 0.0)

        assert xj == x

if __name__ == '__main__':
    IMP.test.main()
