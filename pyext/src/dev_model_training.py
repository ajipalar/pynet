"""
This is a workspace for developing the model_training module.
An dev environment is set up and some elementary tests are performed.
Perhaps these tests will make it into automated testing.
The fact that adding tests isn't super easy is problematic.
Probably I should be building IMP out of tree
"""

import model_training as mt
import model_proto as mp

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from functools import partial


a = jnp.array([0, 0, 1])
len_a = len(a)
flip_probs = jnp.ones(len_a) * 0.5
rseed = 13
key = jax.random.PRNGKey(rseed)
keys_a = jax.random.split(key, len_a)

flippy = partial(mp.flip_edges, len_edge_vector=len_a)
jit_flippy_a = jax.jit(flippy)


def test_move_model():
    pe = mp.get_possible_edges(44)
    mt.move_model(key, mt.M, 0.5, 10, 43, pe, 44) 


def do_tests_dev_model_training():
    test_move_model()



    





