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


class Tests(IMP.test.TestCase):
    def test_magnitude(self):
        a = 1
        b = 2
        example._magnitude(a, b)


if __name__ == "__main__":
    IMP.test.main()
