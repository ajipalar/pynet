from __future__ import print_function
import IMP.test
import IMP.algebra
try:
    import IMP.pynet
except ModuleNotFoundError:
    pass
import io
import os
import math

from .testdefs import example
from hypothesis import given, settings, strategies as st
import unittest

class PropertyTests(IMP.test.TestCase):

    @unittest.expectedFailure
    @given(st.floats(), st.floats())
    def test_a_lt_b(self, a, b):
        example._a_lt_b(a, b)

    @unittest.expectedFailure
    @given(st.floats(), st.floats())
    def test_a_lt_b2(self, a, b):
        example._a_lt_b(a, b)

    @given(st.floats(max_value = 0), st.floats(min_value = 1))
    def test_a_lt_b3(self, a, b):
        example._a_lt_b(a, b)


if __name__ == '__main__':
    IMP.test.main()
