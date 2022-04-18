from __future__ import print_function
import IMP.test
import IMP.algebra
import os
import math
import collections
import jax

from ..src import poissonsqr as td
import pyext.src.poissonsqr as src_module


class Tests(td.PoissUnitTests):
    """Derived class for Poisson SQR Model unit tests"""

    src = src_module
    rtol = 1e-5
    atol = 1e-5
    decimal = 7
    rseed = 13
    key = jax.random.PRNGKey(rseed)

    kwds = collections.namedtuple("KWDS", ["rtol", "atol", "decimal"])(
        rtol, atol, decimal
    )


if __name__ == "__main__":
    IMP.test.main(verbosity=2)
