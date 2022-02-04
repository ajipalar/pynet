#!/usr/bin/env python
# coding: utf-8

import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.lax import fori_loop as jfori
from functools import partial
import pyext.src.plot as plot
from typing import Any
from sklearn.metrics import roc_curve, precision_recall_curve
import termplotlib as tpl

State = Any
Output = Any
KeyArray = Any
DeviceArray = Any

# In[39]:

class StateLess:
    def stateless_method(state: State, *args, **kwargs) -> (State, Output):
        pass

# Funcitonal implementation
def cond_norm(key, y, rho, *args, **kwargs):
    return jax.random.normal(key) * jnp.sqrt(1 - rho **2) + y * rho

def cond_bern(key, p, shape):
    return jax.random.bernoulli(key, p, shape)

def cond_dist(key, dist, params, **kwargs):
    return dist(key, *params, **kwargs)

# X and Y are independant
def gibbsf(key: KeyArray, *args, N=10, thin=10,
           rho=0.5,
           **kwargs) -> DeviceArray:
    
    def outer_body_fun(i: int, val):
        key, samples, x, y, rho, thin  = val
        key, samples, x, y, rho, thin = jax.lax.fori_loop(0, thin, inner_body_fun, val)
        samples = samples.at[i].set([x, y])
        return key, samples, x, y, rho, thin

    def inner_body_fun(i: int, 
                       init_val
                       ) -> tuple[KeyArray, float, float]:
        
        key, samples, x, y, rho, thin = init_val
        key, k1, k2 = jax.random.split(key, 3)
        x = jax.random.normal(key) * jnp.sqrt(1 - rho **2) + y * rho
        y = jax.random.normal(key) * jnp.sqrt(1 - rho **2) + x * rho
        
        return key, samples, x, y, rho, thin
    
    # initiate
    key, k1, k2 = jax.random.split(key, 3)
    x = jax.random.uniform(k1)
    y = jax.random.uniform(k2)
    samples = jnp.zeros((N, 2))
    val = key, samples, x, y, rho, thin
    
    # key, samples, x, y, rho, thin
    samples = jax.lax.fori_loop(0, N, outer_body_fun, val)[1]
    return samples

# imperative implementation

def gibbs(key, N=10, thin=10):
    key, k1, k2 = jax.random.split(key, 3)
    x = jax.random.uniform(k1)
    y = jax.random.uniform(k2)
    samples = jnp.array(np.zeros((N, 2)))
    for i in range(N):
        for j in range(thin):
            key, k1, k2 = jax.random.split(key, 3)
            x = x_cond_dens(k1, y)
            y = y_cond_dens(k2, x)
        samples = samples.at[i].set([x, y])
    return samples


def jit_compile_gibbsf():
    gibbsf_partial = partial(gibbsf, N=5000, thin=100, rho=-0.7)
    gibbsf_jit = jax.jit(gibbsf_partial)
    return gibbsf_jit

def do_gibbsf_sampling():
    gibbsf_jit = jit_compile_gibbsf()
    key = jax.random.PRNGKey(5)
    samples = np.array(gibbsf_jit(key))
    return samples




# Metropolis Hastings

def mh_f(key, steps):
    
    def score(x):
        return jax.scipy.stats.logistic.pdf(x)

    def rv(key):
        return jax.random.uniform(key, minval=-10, maxval=10)

    def body(i, init):
        key, x = init
        x0 = x[i-1]
        key, g1, g2 = jax.random.split(key, 3)
        x1 = rv(g1)
        A = score(x1) / score(x0)
        p = jax.random.uniform(g2)
        xi = jax.lax.cond(p < A, lambda : x1, lambda : x0)
        x = x.at[i].set(xi)
        return key, x
    
    def silly(i, v):
        u, x = v
        w = u, x
        return w
    
    samples = jnp.zeros(steps)
    key, samples = jax.lax.fori_loop(1, steps, body, (key, samples))
    return samples


def local_block():
    def a(state: tuple[Any], x, y, *args, **kwargs):
        return state, (x + y, )

    def b(state: tuple[Any], x, *args, **kwargs):
        return state, (x, x ** 2 )

    def c(state: tuple[Any], x, y, *args, **kwargs):
        return state, (x + y, ), args, kwargs

    def d(state: tuple[Any], x, rho, *args, **kwargs):
        return state, (x - 10 * rho,)
    
    def e(state: tuple[Any], x, *args, **kwargs):
        return state, (x * 2, )
    
    def f(state: tuple[Any], x, *args, **kwargs):
        return state, (jnp.log(x), )
    
    def h(state: tuple[Any], x, *args, **kwargs):
        return state, (1 / 2 * x ** 2, )
    
    # We wish to compose and jit compile these functions into a pipe
    # output = h(e(d(c(b(a)))))
    # each function has positional[:tuple] and keyword[:dict] arguments
    # we would like to make a change in the middle of the chain without
    # affecting how we built the funciton before and after
    # how would we do this?
    
    # supposing the funcitons are pure
    
    # we must make a change to c
    # option1 : add an extra parameter to the highest level function
    # and a parameter to c without changing c's return type
    
    # c(x, y) -> c(x, y, z)
    # 
    
    def statless_function(state: State, *args, **kwargs) -> tuple[State, Output]:
        pass
    
    def pipe(state: tuple[Any], pipe_in, *args, **kwargs):
        state, return_a = a(state, *pipe_in)
        state, return_b = b(state, *return_a)
        state, return_c = c(state, *return_b)
        state, return_d = d(state, *return_c)
        state, return_e = e(state, *return_d)
        state, return_f = f(state, *return_e)
        state, pipe_out = h(state, *return_f)
        return state, pipe_out
    
    class P:
        def __lt__(self, div):
            print(div)
    
    state = (1, 2)
    pipe_in = (11, 12)
    p = P()
    p < pipe(state, pipe_in)
