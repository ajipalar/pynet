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
from .source import ais_testdefs as td


deadline_ms = 2000

class DevTestAIS(IMP.test.TestCase):
    """Test the various functions in the AIS module""" 

    rtol = 1e-05
    atol = 1e-05

    def test_sample(self):
        td.sample(n_samples=5, n_inter=5, decimal_tolerance=5)

    def test_nsteps_mh__g(self):
        mu = 100
        sigma = 2
        rseed = 3
        td.nsteps_mh__g(mu, sigma, rseed)
        
if __name__ == '__main__':
    IMP.test.main()
