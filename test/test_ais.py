from __future__ import print_function
import IMP.test
import IMP.algebra
try:
    import IMP.pynet
    import IMP.pynet.ais as ais
except ModuleNotFoundError:
    import pyext.src.ais as ais

import io
import os
import math


class TestAIS(IMP.test.TestCase):
    """Test the various functions in the AIS module""" 

    def test_magnitude(self):
        """Write the test cast, print statemetns ok"""
        pass




if __name__ == '__main__':
    IMP.test.main()
