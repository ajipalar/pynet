from __future__ import print_function
import IMP.test
import IMP.algebra

try:
    import IMP.pynet
except ModuleNotFoundError:
    import pyext.src
import io
import os
import math


class Tests(IMP.test.TestCase):
    module = IMP.pynet

    def test_magnitude(self):
        """Write the test cast, print statemetns ok"""
        pass

    def test_function_names(self):
        exceptions = []
        words = []
        self.assertFunctionNames(self.module, exceptions, words)

    def test_class_names(self):
        exceptions = []
        words = []
        self.assertClassNames(self.module, exceptions, words)


if __name__ == "__main__":
    IMP.test.main()
