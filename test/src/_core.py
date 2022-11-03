from __future__ import print_function
import IMP
import IMP.test
import numpy as np
import jax.numpy as jnp
from typing import Any
from functools import partial
from collections import namedtuple

Module = Any

def helper_setup():
    p = 5
    Atrue = jnp.zeros((p, p))

    rseed = 13
    mu = 7
    cov = 2
    reps = 3
    data = jax.random.multivariate_normal(key, mu, cov, shape=(reps,))

    g0 = (0, 1, 2, 3, 4)
    g1 = (1, 2, 3, 4)

    Setup = namedtuple("Setup",
                       "p Atrue rseed mu cov reps data g0 g1")

    return Setup(p, Atrue, rseed, mu, cov, reps, data, g0, g1)



def test_model_template_1(src: Module):

    s = helper_setup()
    m = src.ModelTemplate() 
    m.add_node_indices(g0)
    m.add_node_group(g1)
    m.add_node_group(g0)





class CoreTests(IMP.test.TestCase):
    def test_potentially_connected(self):
        test_is_potentially_connected(self.src)

    def test_adjacent_nodes(self):
        test_adjacent_nodes(self.src)

    def test_dfs(self):
        test_dfs(self.src)
