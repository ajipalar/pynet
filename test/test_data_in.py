
from __future__ import print_function
import IMP.test
import IMP.algebra
import IMP.pynet
import io
import os
import math
try:
    import IMP.pynet.data_in as di
except:
    import pyext.src.data_in as di

class TestDataIn(IMP.test.TestCase):

    synthetic_data = Path('pyext/data/synthetic/41586_2020_2286_MOESM5_ESM.csv')

    def test_magnitude(self):
        """Write the test cast, print statemetns ok"""
        pass

    def test_read_column_n():
        """ tests data input """
        for column in range(0, 12):

        pass


if __name__ == '__main__':
    IMP.test.main()
