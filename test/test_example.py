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

from .src import example


class DevTests(IMP.test.TestCase):
    def test_a_lt_b(self):
        a = 1
        b = 2
        example._a_lt_b(a, b)


if __name__ == "__main__":
    IMP.test.main()
