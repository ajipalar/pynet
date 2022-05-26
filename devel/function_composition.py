# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import jax
import numpy as np
import jax.numpy as jnp
import itertools
import timeit

a = lambda x: x / 13
b = lambda x: 2 + x - 1
c = lambda x: 2**x
d = lambda x: jnp.log(x)
e = lambda x: x*jnp.pi
f = lambda x: x - 0.5*x
g = lambda x: x % 190
h = lambda x: g(f(e(d(c(b(a(x)))))))

h_jit = jax.jit(h)

# %timeit h(99)

# %timeit h_jit(99).block_until_ready()

l = [1, 2, 3, 4, 5, 6, 7]
h_stream = map(h, l)

# %timeit list(map(h, l))

# %timeit list(map(h_jit, l))

z = lambda x: d(e(x))
z = jax.jit(z)

z(7)
