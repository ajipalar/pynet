"""
Use from workspace import * from an active python session

"""

from functools import partial
from itertools import combinations, permutations
import jax
from jax import jit, grad, vmap, make_jaxpr, value_and_grad
from jax.lax import scan, cond, fori_loop, while_loop
import jax.numpy as jnp
import jax.scipy as jsp
from jax.tree_util import tree_flatten, tree_unflatten
import numpy as np
import pandas as pd
import scipy as sp
import scipy.special
import operator as op
import pyile


# Possibly unstable
from tensorflow_probability.substrates import jax as tfp

# Global User Variables
rseed = 13
key = jax.random.PRNGKey(rseed)
d = 5
A = jax.random.bernoulli(key, shape=(d, d))
A = A.astype(jnp.int32).block_until_ready()
L0 = jnp.tril(A)
U0 = jnp.triu(A)


# Testing the multivariate pdf without a constrained matrix
x = jnp.array([0, 1, 2, 1, 0])
mean = jnp.array([1, 1, 1, 1, 0])

B = jnp.arange(5*5).reshape((5, 5))

