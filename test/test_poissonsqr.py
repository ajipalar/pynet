from __future__ import print_function
import IMP.test
import IMP.algebra
import os
import math

from . import src.poissonsqr as td
import  IMP.pynet.poissonsqr as src_module

class Tests(td.PoissUnitTests):
    """Derived class for Poisson SQR Model unit tests"""
    src = src_module
    rtol=1e-5
    atol=1e-5
    decimal = 7

    kwds = collections.namedtuple(
            'KWDS', ['rtol', 'atol', 'decimal'])(rtol, atol, decimal)



if __name__ == '__main__':
    IMP.test.main()
