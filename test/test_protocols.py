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

from .src import protocols_testdefs as td


class Tests(IMP.test.TestCase):
    def test_magnitude(self):
        ...


if __name__ == "__main__":
    IMP.test.main()
