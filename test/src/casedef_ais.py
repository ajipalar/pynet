from __future__ import print_function
import IMP.test
import IMP.algebra

import io
import jax
import jax.numpy as jnp
import math
import numpy as np
from functools import partial 
from hypothesis import given, settings, strategies as st
from test.src import ais as td


deadline_ms = 2000

class DevPropertyTrivial(IMP.test.TestCase):
    dist = None # the base class should override this attribute
    ais = None
    

    # Run functions once to jit compile
    def test_run_once(self):
        td.trivial_is_get_invariants_jittable(1, 2, self.dist)
        td.trivial_is_s_rv_jittable(1, 2, self.dist)
        td.trivial_is_T_jittable(1, 2, self.dist)
        td.trivial_is_get_log_score_jittable(1, 2, self.dist)
        

    @given(st.integers(min_value=1, max_value=2147483647), st.integers(min_value=1, max_value=2147483647))
    def test_trivial(self, n1, n2):
        td.trivial_is_get_invariants_jittable(n1, n2, self.dist)
    
        td.trivial_is_s_rv_jittable(n1, n2, self.dist)
    
        td.trivial_is_T_jittable(n1, n2, dist)
    
        td.trivial_is_get_log_score_jittable(n1, n2, self.dist)


class DevTrivialAIS(IMP.test.TestCase):
    dist = None
    ais = None

    n1=1
    n8 = 8
    n64 = 64
    n512 =512
    n4k=4096
    n32k=32768

    seed1=111
    seed2=1234


    def test_model_getter1(self):
        td.trivial_is_get_invariants_jittable(self.n1, self.n1, self.dist)

    def test_s_rv1(self):
        td.trivial_is_s_rv_jittable(self.n1, self.n1, self.dist)

    def test_T1(self):
        td.trivial_is_T_jittable(self.n1, self.n1, self.dist)

    def test_log_score1(self):
        td.trivial_is_get_log_score_jittable(self.n1, self.n1, self.dist)

    def test_model_getter8(self):
        td.trivial_is_get_invariants_jittable(self.n8, self.n8, self.dist)

    def test_s_rv8(self):
        td.trivial_is_s_rv_jittable(self.n8, self.n8, self.dist)

    def test_T8(self):
        td.trivial_is_T_jittable(self.n8, self.n8, self.dist)

    def test_log_score8(self):
        td.trivial_is_get_log_score_jittable(self.n8, self.n8, self.dist)

    def test_model_getter64(self):
        td.trivial_is_get_invariants_jittable(self.n64, self.n64, self.dist)

    def test_s_rv64(self):
        td.trivial_is_s_rv_jittable(self.n64, self.n64, self.dist)

    def test_T64(self):
        td.trivial_is_T_jittable(self.n64, self.n64, self.dist)

    def test_log_score64(self):
        td.trivial_is_get_log_score_jittable(self.n64, self.n64, self.dist)

    def test_model_getter512(self):
        td.trivial_is_get_invariants_jittable(self.n512, self.n512, self.dist)

    def test_s_rv512(self):
        td.trivial_is_s_rv_jittable(self.n512, self.n512, self.dist)

    def test_T512(self):
        td.trivial_is_T_jittable(self.n512, self.n512, self.dist)

    def test_log_score512(self):
        td.trivial_is_get_log_score_jittable(self.n512, self.n512, self.dist)

    def test_model_getter4k(self):
        td.trivial_is_get_invariants_jittable(self.n4k, self.n4k, self.dist)

    def test_s_rv4k(self):
        td.trivial_is_s_rv_jittable(self.n4k, self.n4k, self.dist)

    def test_T4k(self):
        td.trivial_is_T_jittable(self.n4k, self.n4k, self.dist)

    def test_log_score4k(self):
        td.trivial_is_get_log_score_jittable(self.n4k, self.n4k, self.dist)

    def test_model_getter32k(self):
        td.trivial_is_get_invariants_jittable(self.n32k, self.n32k, self.dist)

    def test_s_rv32k(self):
        td.trivial_is_s_rv_jittable(self.n32k, self.n32k, self.dist)

    def test_T32k(self):
        td.trivial_is_T_jittable(self.n32k, self.n32k, self.dist)

    def test_log_score32k(self):
        td.trivial_is_get_log_score_jittable(self.n32k, self.n32k, self.dist)

    def test_specialized_model_to_sampling_trivial(self):
        td.specialize_model_to_sampling_trivial(
            n_samples=8, n_inter=8, decimals=5, ais=self.ais, dist = self.dist)

    
    def test_negative_sample_trivial(self):
        td.negative_sample_trivial(
            n_samples=self.n512,
            n_inter=self.n1,
            rseed1=self.seed1,
            rseed2=self.seed2,
            ais=self.ais,
            dist=self.dist)

class DevTrivialBetaDependantAIS(IMP.test.TestCase):
    dist = None
    ais = None
    ...




class DevTestAIS(IMP.test.TestCase):
    """Test the various functions in the AIS module""" 
    dist = None
    ais = None
    

    rtol = 1e-05
    atol = 1e-05

    n_samples8=8
    n_inter8=8

    n_samples64=64
    n_inter64=64

    dist = None




    @IMP.test.skip
    def test_sample(self):
        td.sample(n_samples=5, n_inter=5, decimal_tolerance=5)

    @IMP.test.skip
    def test_nsteps_mh__g(self):
        td.nsteps_mh__g(mu=100, sigma=2, rseed=3)
        
if __name__ == '__main__':
    IMP.test.main()
