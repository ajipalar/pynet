import jax
import jax.numpy as jnp
import jax.scipy as jsp
import pyext.src.lpdf as lpdf
import pyext.src.lpmf as lpmf
import pyext.src.matrix as mat
import pyext.src.pynet_rng as rng
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, permutations
from functools import partial
from collections import namedtuple
from jax.tree_util import tree_flatten, tree_unflatten

from ToyModelGlobals import (inv, split, SampleParams, SampleState, InfState)
from ToyModel0 import (
        init_inference_state,
        dosaves
)

from ToyModelSampling import (
        init_sampling_state,
        dontsave,
        core_update,
        do_inference
)

import ToyModel0 as model_0
import ToyModelSampling as sampling

key = jax.random.PRNGKey(12983)
sample_params = SampleParams(nmc_steps=1000,
                            step=0,
                            T=2,
                            key=key,
                            accepted=False)

sample_state = init_sampling_state(model_0.position, sample_params)
val = sample_state, init_inference_state(1000, 100)
core_update(False, val) 
inference_loop = partial(do_inference, update_fun=update_sampling_state,
                        nsteps=sample_params.nmc_steps, save_position_every=100)

