from __future__ import print_function
import IMP.test
import IMP.algebra
import os
import math
import collections
import jax
import jax.numpy as jnp
import numpy as np

from ..src import graph as td
import pyext.src.graph as src_module

class Tests(td.GraphTests):
    """Derived class for graph module"""

    src = src_module
    rtol = 1e-5
    atol = 1e-5
    decimal = 7
    rseed = 13
    key = jax.random.PRNGKey(rseed)

    kwds = collections.namedtuple("KWDS", ["rtol", "atol", "decimal"])(
        rtol, atol, decimal
    )

    # Stateful Test Information for 2d tests

if __name__ == "__main__":
    IMP.test.main(verbosity=2)
