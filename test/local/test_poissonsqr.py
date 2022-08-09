from __future__ import print_function
import IMP.test
import IMP.algebra
import os
import math
import collections
import jax
import jax.numpy as jnp
import numpy as np

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

    # Stateful Test Information for 2d tests

    theta2d = np.array([-1.0, 1.0])
    phi2d = np.array(
        [[1.0, 0.0], [0.0, 1.0]],
    )

    x2d = np.array([0, 0], dtype=jnp.int32)
    get_exp2d__j = src_module.get_exponent__s(theta=theta2d, phi=phi2d, x=x2d)
    get_exp2d = get_exp2d__j  # jit and hypothesis-numpy are not friends


if __name__ == "__main__":
    IMP.test.main(verbosity=2)
