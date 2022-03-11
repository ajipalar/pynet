from __future__ import print_function
import IMP.test
import IMP.algebra
try:
    import IMP.pynet
    import IMP.pynet.ais as ais
    import IMP.pynet.functional_gibbslib as fg
    import IMP.pynet.PlotBioGridStatsLib as bsl
    import IMP.pynet.distributions as dist
except ModuleNotFoundError:
    import pyext.src.ais as ais
    import pyext.src.functional_gibbslib as fg
    import pyext.src.PlotBioGridStatsLib as nblib
    import pyext.src.distributions as dist

import io
import jax
import jax.numpy as jnp
import math
import numpy as np
from functools import partial 
from typing import Union, Any, Callable



def nsteps_mh__g(mu : float, sigma: float, rseed : Union[float, int]):
    log_intermediate__j = partial(dist.norm.lpdf, loc=mu, scale=sigma)
    n_steps = 100

    key = jax.random.PRNGKey(rseed)
    x = 0.0

    kwargs_nsteps_mh = {
            'log_intermediate__j': log_intermediate__j,
            'intermediate_rv__j': dist.norm.rv,
            'n_steps': n_steps,
            'kwargs_log_intermediate__j': {}
    }
            

    #Callable
    x = ais.nsteps_mh__g(key, x, **kwargs_nsteps_mh)

    #Jittable
    nsteps_mh__j = partial(ais.nsteps_mh__g, **kwargs_nsteps_mh)
    nsteps_mh = jax.jit(nsteps_mh__j)

    xj = nsteps_mh(key, 0.0)

    assert xj == x

def nsteps_mh__g_accuracy(mu, cv):

    if mu == 0:
        sigma = cv
    else:
        sigma = jnp.abs(mu * cv)

    log_intermediate__j = partial(dist.norm.lpdf, loc=mu, scale=sigma)

    key = jax.random.PRNGKey(7)
    n_steps = 50000

    kwargs_nsteps_mh = {
            'log_intermediate__j': log_intermediate__j,
            'intermediate_rv__j': dist.norm.rv,
            'n_steps': n_steps,
            'kwargs_log_intermediate__j': {}
    }

            

    #Jittable
    nsteps_mh__j = partial(ais.nsteps_mh__g, **kwargs_nsteps_mh)
    nsteps_mh = jax.jit(nsteps_mh__j)

    xj = nsteps_mh(key, 0.0)
    assert xj > mu - 4*sigma
    assert xj < mu + 4*sigma

     

def apply_normal_context_to_sample(mu : float, sigma : float, 
        n_mh_steps  : int, n_samples : int, n_inter : int, rseed):

    f = ais.apply_normal_context_to_sample__s
    sample__j = f(mu, sigma, n_mh_steps, n_samples, n_inter)
    #sample = jax.jit(sample__j)
    key = jax.random.PRNGKey(rseed)

    weights, samples, = sample__j(key)

def f0_pdf__j(mu, sig):

    cases1 = [mu -1, mu, mu + 1]
    cases2 = [mu - 10, mu, mu + 10]

    f0 = ais.f0_pdf__j
    f0 = partial(f0, mu = mu, sig = sig)
    jf0 = jax.jit(f0)
    jf0(2.0).block_until_ready()

    assert f0(cases1[0]) > f0(cases2[0])
    assert f0(cases1[1]) == f0(cases2[1])
    assert f0(cases1[2]) > f0(cases2[2])

    for n in jnp.arange(-100, 100):
        assert 0 <= f0(n) <= 1
        np.testing.assert_almost_equal(jf0(n), f0(n), decimal = 5)

def fn_pdf__j(mu, sig):
    
    cases1 = [mu -1, mu, mu + 1]
    cases2 = [mu - 10, mu, mu + 10]

    f0 = ais.fn_pdf__j
    print('class implementation')
    f0 = ais.norm.pdf
    jf0 = jax.jit(f0)
    jf0(2.0).block_until_ready()

    assert f0(cases1[0]) > f0(cases2[0])
    assert f0(cases1[1]) == f0(cases2[1])
    assert f0(cases1[2]) > f0(cases2[2])

    for n in jnp.arange(-100, 100):
        assert 0 <= f0(n) <= 1
        np.testing.assert_almost_equal(jf0(n), f0(n), decimal = 5)

def fj_pdf__g(mu, sig):

    target__j = ais.f0_pdf__j
    target__j = partial(target__j, mu = mu, sig = sig)
    source__j = ais.fn_pdf__j

    fj_pdf__j = partial(ais.fj_pdf__g, target__j = target__j, source__j = source__j)

    jfj_pdf = jax.jit(fj_pdf__j)
    jfj_pdf(2.0, 0.3).block_until_ready()

    for beta in jnp.arange(0, 1, 0.1):
        ...





def T_nsteps__unorm2unorm__p(mu, sig):

    T_j = ais.T_nsteps_mh__unorm2unorm__p(mu, sig)
    T_j = ais.T.unorm2unorm__p(mu, sig)
    print('class encapsulation')

    x = jnp.arange(0, 20)
    betas = jnp.arange(0, 1, 0.1)

    maximum = 0.0

    key = jax.random.PRNGKey(1)

    for i, xi in enumerate(x):
        for j, bij in enumerate(betas):
            key, subkey = jax.random.split(key, 2)
            xij = T_j(key, xi, kwargs_intermediate__j = {'beta': bij})


def test_T_nsteps_mh__g(rseed, x):
    key = jax.random.PRNGKey(rseed)
    ij = ais.fj_pdf__g
    ij__j = partial(ij, source__j = ais.fn_pdf__j,
            target__j = ais.f0_pdf__j)

    irvs = jax.random.normal
    kwargs = {}

    test_func = ais.T_nsteps_mh__g
    kwargs_test_func = {'key': key, 'x': x, 'intermediate_rv__j' : irvs,
            'intermediate__j': ij,  'kwargs_intermediate__j' : kwargs}

    test_func__j = partial(test_func, intermediate__j = ij__j) 

    jresults = jax.jit(test_func)(**kwargs_test_func)
    assert jresults == test_func(**kwargs_test_func)


def do_ais(mu, sigma):
    fn_pdf = jax.scipy.stats.norm.pdf
    n_samples = 100
    n_inter = 50
    betas = np.linspace(0, 1, n_inter)
    key = jax.random.PRNGKey(10)
    T = ais.T_nsteps_mh__g

    fj_pdf__j = partial(ais.fj_pdf__g, source__j = jax.scipy.stats.norm.pdf,
            target__j = ais.f0_pdf__j)





    samples, weights = ais.do_ais__g(key = key, 
                                  mu = mu,
                                  sigma = sigma,
                                  n_samples = n_samples, 
                                  n_inter = n_inter, 
                                  betas = betas,
                                  f0_pdf = ais.f0_pdf__j,
                                  fj_pdf = fj_pdf__j,
                                  fn_pdf = ais.fn_pdf,
                                  transition_rule__j = T)

    atol_l = [10 ** i for i in range(1, -6, -1)]
    rtol_l = [i * 10 ** -1 for i in range(9, 1, -1)]

    mean = ais.get_mean(samples, weights)
    for rtol in rtol_l:
        np.testing.isclose(mean, mu, rtol = rtol)
