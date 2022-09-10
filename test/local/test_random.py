from __future__ import print_function
import IMP.test
import IMP.algebra
import os
import math
import collections
import jax
import jax.numpy as jnp
import numpy as np

from ..src import random as td

import pyext.src.random as src_module


class Tests(td.RandomUnitTests):
    ...


if __name__ == "__main__":
    IMP.test.main(verbosity=2)
