from __future__ import print_function
import IMP.test
import IMP.algebra
try:
    import IMP.pynet.ais as ais
except ModuleNotFoundError:
    import pyext.src.ais as ais

import io
import os
import math
from math import isnan
from hypothesis import assume, given, settings, strategies as st
from .source import ais_testdefs as td

class NegativeTestAIS(IMP.test.TestCase):
    @given(st.integers(), st.integers())
    def test_negative_sample_trivialI(self, seed1: int, seed2: int):
        assume(not isnan(seed1))
        assume(not isnan(seed2))

        td.negative_sample_trivial(
            n_samples=50,
            n_inter=2,
            rseed1=seed1,
            rseed2=seed2
        )

    @IMP.test.skip
    @given(st.integers(min_value=1, max_value=1000))
    def test_negative_sample_trivial2(self, n_samples):
        assume(not isnan(n_samples))

        td.negative_sample_trivial(
            n_samples=n_samples,
            n_inter=2,
            rseed1=111,
            rseed2=112)


if __name__ == '__main__':
    IMP.test.main()
