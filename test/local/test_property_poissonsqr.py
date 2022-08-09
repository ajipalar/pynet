from __future__ import print_function
import IMP.test
import IMP.algebra
import os
import math
import collections

from ..src import poissonsqr as td
import pyext.src.poissonsqr as src_module
import jax.numpy as jnp
import jax
import numpy as np


class Tests(td.PoissPropTests):
    """Derived class for Poisson SQR Model property unit tests"""

    src = src_module
    rtol = 1e-5
    atol = 1e-5
    decimal = 7

    kwds = collections.namedtuple("KWDS", ["rtol", "atol", "decimal"])(
        rtol, atol, decimal
    )

    # Stateful Test Information for 2d tests

    theta2d = np.array([-1.0, 1.0])
    phi2d = np.array(
        [[1.0, 0.0], [0.0, 1.0]],
    )

    x2d = np.array([0, 0], dtype=jnp.int32)
    get_exp2d__j = src_module.get_exponent__s(theta=theta2d, phi=phi2d, x=x2d)
    get_exp2d = get_exp2d__j  # jit and hypothesis-numpy are not friends
#    get_exp2d = jax.jit(get_exp2d__j)
#    get_exp2d(theta=theta2d, phi=phi2d, x=x2d, i=0)
    
    


if __name__ == "__main__":
    IMP.test.main(verbosity=2)
