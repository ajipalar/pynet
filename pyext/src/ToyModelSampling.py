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



def init_sampling_state(position, sample_params):
    return SampleState(position, sample_params)
    

# Sample alternative models

# Report

# +

def dontsave(val):
    state, infstate = val
    step = infstate.step
    step = step + 1
    scores = infstate.scores
    accepteds = infstate.accepteds
    As = infstate.As
    Sigma1s = infstate.Sigma1s
    Sigma2s = infstate.Sigma2s
    saveindex = infstate.saveindex

    infstate = InfState(scores, accepteds, As, Sigma1s, Sigma2s, step, saveindex)
    val = state, infstate
    return val


def core_update(notsave, val):
    return jax.lax.cond(notsave==False,
                       dosaves,
                       dontsave,
                       val)

def general_inference_body(i, val, update_fun):
    state, infstate, nsaves = val
    step = infstate.step
    notsave = step % nsaves
    
    val = state, infstate

    scratch, updated_infstate = core_update(notsave, val)

    updated_state = update_fun(state)
    val = updated_state, updated_infstate, nsaves
    return val

def do_inference(init_state, update_fun, nsteps, save_position_every):

    inference_body = partial(general_inference_body, update_fun=update_fun)
    
    # save the score, position, every save_every
    scores, accepteds, nsaves, As, Sigma1s, Sigma2s = init_inference(nsteps, save_position_every)
    init_infstate = InfState(scores, accepteds, As, Sigma1s, Sigma2s, 0, 0)
    

    val = init_state, init_infstate, nsaves
    return jax.lax.fori_loop(0, nsteps, inference_body, val)
