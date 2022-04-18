from .typedefs import PRNGKeyArray, DeviceArray, Dimension, Output, State

from . import plot as plot

import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.lax import fori_loop as jfori
from functools import partial
from typing import Any, Callable
from sklearn.metrics import roc_curve, precision_recall_curve
import inspect
import re


class StateLess:
    def stateless_method(state: State, *args, **kwargs) -> (State, Output):
        pass


# Funcitonal implementation
def cond_norm(key, y, rho, *args, **kwargs):
    return jax.random.normal(key) * jnp.sqrt(1 - rho ** 2) + y * rho


def cond_bern(key, p, shape):
    return jax.random.bernoulli(key, p, shape)


"""
def cond_bern(key, i, state, args):
    x = jax.random.bernoulli(key, args)
    state = state.at[i].set(x)
    return state
"""


def cond_dist(key, dist, params, **kwargs):
    return dist(key, *params, **kwargs)


# X and Y are independant


def gibbsf(
    key: PRNGKeyArray, *args, N=10, thin=10, rho=0.5, dof=2, **kwargs
) -> DeviceArray:
    def outer_body_fun(i: int, val):
        key, samples, params, rho, thin = val
        key, samples, params, rho, thin = jax.lax.fori_loop(
            0, thin, inner_body_fun, val
        )
        samples = samples.at[i].set(params)
        return key, samples, params, rho, thin

    def inner_body_fun(i: int, init_val) -> tuple[PRNGKeyArray, float, float]:

        key, samples, params, rho, thin = init_val
        key, k1, k2 = jax.random.split(key, 1 + dof)
        # x = jax.random.normal(k1) * jnp.sqrt(1 - rho **2) + y * rho
        # y = jax.random.normal(k2) * jnp.sqrt(1 - rho **2) + x * rho
        x = jax.random.bernoulli(k1, 0.5)
        y = jax.random.bernoulli(k2, 0.5)

        params = [x, y]
        return key, samples, params, rho, thin

    # initiate
    key, k1, k2 = jax.random.split(key, 3)
    # x = jax.random.uniform(k1)
    # y = jax.random.uniform(k2)

    x = jax.random.bernoulli(k1)
    y = jax.random.bernoulli(k2)

    samples = jnp.zeros((N, 2))
    params = [x, y]
    val = key, samples, params, rho, thin

    # key, samples, x, y, rho, thin
    samples = jax.lax.fori_loop(0, N, outer_body_fun, val)[1]
    return samples


# In[3]:


def generic_gibbsf(
    key: PRNGKeyArray,
    nsamples: int,
    thin_every: int,
    nparams: Dimension,
    init_params: Callable,
    update_params: Callable,
    init_args=[],
    init_kwargs={},
    update_args=[],
    update_kwargs={},
) -> DeviceArray:
    """Generic version of gibbs sampling, partial application with known nsamples,
    nparams, thin_every, init_params, update_params, yields a jit compilable gibbs sampler

    params:
      key : PRNGKey
      nsamples : int , total number of samples. Final number of samples will be nsamples / thin_every
      thin_every : the Markov chain thinning rate. Save a sample every thin steps
      init_params : the initial generating function for the samples. Takes a key parameter
      update_params : A gibbs sequential update function
    returns:
      samples : an (N / thin_every, nparams) dimensional DeviceArray of samples
    """

    def outer_body_fun(
        i: int, val
    ) -> tuple[PRNGKeyArray, PRNGKeyArray, int, DeviceArray, DeviceArray]:

        thin_every = val[2]

        key, k1, thin_every, params, samples = jax.lax.fori_loop(
            0, thin_every, inner_body_fun, val
        )
        samples = samples.at[i].set(params)

        return key, k1, thin_every, params, samples

    def inner_body_fun(
        i: int, init_val
    ) -> tuple[PRNGKeyArray, PRNGKeyArray, int, DeviceArray, DeviceArray]:

        key, k1, thin_every, params, samples = init_val
        key, k1 = jax.random.split(k1)
        params = update_params(key, params, *update_args, **update_kwargs)

        return key, k1, thin_every, params, samples

    # initiate
    key, k1 = jax.random.split(key, 2)
    params = init_params(key, *init_args, **init_kwargs)
    samples = jnp.zeros((nsamples, nparams))

    val = key, k1, thin_every, params, samples

    samples = jax.lax.fori_loop(0, nsamples, outer_body_fun, val)[4]
    return samples


def generic_gibbs(
    key: PRNGKeyArray,
    nsamples: int,
    thin_every: int,
    nparams: Dimension,
    init_params: Callable,
    update_params: Callable,
    init_args=[],
    init_kwargs={},
    update_args=[],
    update_kwargs={},
) -> DeviceArray:

    key, k1 = jax.random.split(key, 2)

    param = init_params(key, *init_args, **update_kwargs)
    samples = jnp.zeros((nsamples, nparams))

    for i in range(nsamples):
        for j in range(thin_every):

            key, k1 = jax.random.split(k1)
            param = update_params(key, param, *update_args, **update_kwargs)

        samples = samples.at[i].set(param)

    return samples


def x_cond_dens(key, m):
    rho = 0.3
    return cond_norm(key, m, rho)


y_cond_dens = x_cond_dens


def example_generic_init_params(key, state=None):
    # print(key)
    k1, k2 = jax.random.split(key)
    x = jax.random.uniform(k1)
    y = jax.random.uniform(k2)
    # print(x, y)
    params = jnp.array([x, y])
    return params


def example_generic_update_params(key, params, state=None):
    k1, k2 = jax.random.split(key, 2)
    x, y = params
    x = x_cond_dens(k1, y)
    y = y_cond_dens(k2, x)
    return jnp.array([x, y])


# imperative implementation
def gibbs(key, N=10, thin=10):
    # print('calling gibbs')
    key, k1 = jax.random.split(key, 2)  # extra call to assert same as generic
    # print(key)
    ka, kb = jax.random.split(key, 2)

    # Initiate x & y

    x = jax.random.uniform(ka)
    y = jax.random.uniform(kb)
    # print(x, y)
    samples = jnp.zeros((N, 2))

    for i in range(N):
        for j in range(thin):
            key, k1 = jax.random.split(k1)
            ka, kb = jax.random.split(key, 2)

            # draw x given y
            x = x_cond_dens(ka, y)
            # draw y given x
            y = y_cond_dens(kb, x)

        # save the sample every thin steps

        samples = samples.at[i].set(jnp.array([x, y]))
    return samples
