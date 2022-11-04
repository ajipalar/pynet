from __future__ import print_function
import IMP
import IMP.test
import numpy as np
import jax.numpy as jnp
from typing import Any
from functools import partial
from collections import namedtuple
import jax

Module = Any

def helper_setup():
    p = 5
    Atrue = jnp.zeros((p, p))

    rseed = 13
    mu = np.ones(p) * 7
    cov = np.eye(p)
    reps = 3
    key = jax.random.PRNGKey(rseed)
    data = jax.random.multivariate_normal(key, mu, cov=cov, shape=(reps,))

    g0 = (0, 1, 2, 3, 4)
    g1 = (1, 2, 3, 4)

    Setup = namedtuple("Setup",
                       "p Atrue rseed mu cov reps data g0 g1")

    return Setup(p, Atrue, rseed, mu, cov, reps, data, g0, g1)



def test_model_template_1(src: Module):

    s = helper_setup()
    m = src.ModelTemplate() 
    m.add_node_indices(s.g0)
    m.add_node_group(s.g1)
    m.add_node_group(s.g0)

def test_add_point(src):
    s = helper_setup()
    m = src.ModelTemplate()
    m.add_point(0)
    assert m.position[0] == {}
    assert len(m.restraints) == 0


def test_add_contiguous_nodes(src):
    s = helper_setup()
    m = src.ModelTemplate()
    m.add_contiguous_nodes(0, 10)
    for i, scope in enumerate(m.position):
        assert i==scope


class CoreTests(IMP.test.TestCase):
    def test_add_point(self):
        test_add_point(self.src)

    def test_model_template_1(self):
        test_model_template_1(self.src)

    def test_add_contiguous_nodes(self):
        test_add_contiguous_nodes(self.src)
