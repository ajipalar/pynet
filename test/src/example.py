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


# unittest test_cases


def _a_lt_b(a, b):
    """Write the test cast, print statemetns ok"""
    assert a < b
