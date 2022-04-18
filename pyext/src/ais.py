from __future__ import print_function

try:
    from IMP.pynet.typedefs import (
        Array,
        DeviceArray,
        Dimension,
        Index,
        JitFunc,
        Matrix,
        Number,
        PartialF,
        PDF,
        lPDF,
        PMF,
        lPMF,
        PRNGKeyArray,
        PureFunc,
        Samples,
        Vector,
        Weights,
        RV,
        fParam,
        iParam,
        Prob,
        lProb,
        GenericInvariants,
    )
    import IMP.pynet.functional_gibbslib as fg
    import IMP.pynet.PlotBioGridStatsLib as nblib
    import IMP.pynet.distributions as dist
except ModuleNotFoundError:
    from pyext.src.typedefs import (
        Array,
        DeviceArray,
        Dimension,
        Index,
        JitFunc,
        Matrix,
        Number,
        PartialF,
        PDF,
        lPDF,
        PMF,
        lPMF,
        PRNGKeyArray,
        PureFunc,
        Samples,
        Vector,
        Weights,
        RV,
        fParam,
        iParam,
        Prob,
        lProb,
        GenericInvariants,
    )
    import pyext.src.functional_gibbslib as fg
    import pyext.src.PlotBioGridStatsLib as nblib
    import pyext.src.distributions as dist

from abc import ABC, abstractmethod
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from typing import Callable

"""This module contains functions that implement Annealed Importance Sampling.
   In particular Annealed Importance Sampling is implemented for the Poisson 
   Square Root Graphical Model as described in Inouye 2016.
   
   In general AIS is implemented using a generic algorithm that is may be jit 
   compiled using jax's jit compiler for use on the XLA kernal.

   The arguments to set_up_ais allow for jit compilation.

   The functions below are implemented as pure functions with no side effects.

Dev note:
    1. Defined generic function (e.g., sample)
    2. Defined lower function signatures (e.g., sample docstring)
    3. Defined a generic specializer (e.g., specialize_model_to_sampling)
    4. Defiend a specific model getter (e.g., get_normal_model)
"""


def specialize_model_to_sampling(
    model_getter: Callable, kwargs_params: dict, n_samples: int, n_inter: int
) -> JitFunc:

    """A generic specializer that composes model and sample
    to yield a jittable ais sampling function"""

    kwargs_dimension = {"n_samples": n_samples, "n_inter": n_inter}

    # No intersecting keys
    assert len(kwargs_params.keys() & kwargs_dimension.keys()) == 0
    kwargs_model_getter = kwargs_params | kwargs_dimension

    packed = partial(model_getter, **kwargs_model_getter)()
    get_invariants, Source, T, get_log_intermediate_score = packed

    kwargs_sample = {
        "get_invariants": get_invariants,
        "source": Source,
        "T": T,
        "get_log_intermediate_score": get_log_intermediate_score,
    }

    assert len(kwargs_sample.keys() & kwargs_dimension.keys()) == 0

    kwargs_sample = kwargs_sample | kwargs_dimension

    sample__j: Callable[[PRNGKeyArray], tuple[Samples, LogWeights]]
    sample__j = partial(sample, **kwargs_sample)
    return sample__j


def get_normal_model(
    mu: float, sigma: float, n_samples: int, n_inter: int
) -> tuple[Callable, object, Callable, Callable]:

    """Get the function necassary to specialized `sample`
    for the univariate normal distributions using the n_steps_mh
    algorithm"""

    def get_invariants(n_samples: Index, n_inter: Index) -> tuple:
        betas = jnp.arange(0, n_inter)
        return betas

    class Source:
        def rv(key: PRNGKeyArray):
            return jax.random.normal(key)

    source = dist.norm  # pass in a module

    def T(key: PRNGKeyArray, x, t, n, sample_state=None):
        return jax.random.uniform(key)

    def get_log_intermediate_score(x, n, sample_state=None):
        return dist.norm.lpdf(x, loc=mu, scale=sigma)

    return get_invariants, source, T, get_log_intermediate_score


class ContractSample(ABC):
    @abstractmethod
    def get_invariants(n_samples: Index, n_inter: Index) -> GenericInvariants:
        ...

    @abstractmethod
    def T(
        key: PRNGKeyArray,
        x: DeviceArray,
        t: Index,
        n: Index,
        invariants: GenericInvariants,
    ) -> DeviceArray:
        ...

    @abstractmethod
    def get_log_intermediate_score(
        x: DeviceArray, n: Index, sample_state: dict
    ) -> float:
        ...


class DerivedTypeViolation(ContractSample):
    def get_invariants(x: float, y: float) -> float:
        return 2.0


def sample(
    key: PRNGKeyArray = None,
    n_samples: Dimension = None,
    n_inter: Dimension = None,
    get_log_intermediate_score: JitFunc = None,
    source=None,
    T: Callable = None,
    get_invariants: Callable = None,
) -> tuple[Samples, Weights]:

    """A generic algorithm for AIS sampling.
       Four objects are required.
       params:

       function signatures:

         get_invariants:
           (Index, Index) -> (GenericInvariants)

         x: DeviceArray. shape, dtype t[s]

         source:
           .rv : (key) -> (t[s])
         T:
           (PRNGKeyArray, DeviceArray, Index, Index, GenericInvariants) -> (DeviceArray)
         get_log_intermediate_score :
           (t[s], Index, Kwargs) -> (log_score)
      return:
        samples:
        log_weights:


    Imperative implementation

    log_weights: Weights = jnp.zeros(n_samples)
    samples: Samples = jnp.zeros(n_samples)

    invariants = get_invariants(n_samples, n_inter)

    # loop signature
    # (key) -> (samples, log_weights)
    for t in range(n_samples):
        # Sample initial point from q(x)
        #x = p_n.rvs() #random variates

        key, s1, s2 = jax.random.split(key, 3)
        x = source.rv(s1) #jax.random.normal(key)
        logw = 0.0

        # loop signature
        # (s2, x, t, n, invariants) -> logw
        for n in range(1, n_inter):
            # Transition
            #x = transition_rule__j(subkey, x, lambda x: intermediate_j(x, betas[n]), n_steps=5)

            s2, s3 = jax.random.split(s2, 2)

            x = T(s3, x, t, n, sample_state=invariants)

            #What about the betas?

            #Compute weight in log space

            logw += get_log_intermediate_score(x, n, sample_state=invariants) - get_log_intermediate_score(x, n-1, sample_state=invariants)

        samples = samples.at[t].set(x)
        log_weights = log_weights.at[t].set(logw)
    """

    def sample_loop(t, val):
        key, invariants, samples, log_weights = val

        key, s1, s2 = jax.random.split(key, 3)
        x = source.rv(s1)  # jax.random.normal(key)
        logw = 0.0

        inter_init = (s2, x, t, logw, invariants)
        inter_return = jax.lax.fori_loop(1, n_inter, inter_loop, inter_init)
        s2, x, t, logw, invariants = inter_return

        samples = samples.at[t].set(x)
        log_weights = log_weights.at[t].set(logw)

        key = s2

        return key, invariants, samples, log_weights

    def inter_loop(n, val):
        s2, x, t, logw, invariants = val

        s2, s3 = jax.random.split(s2, 2)
        x = T(s3, x, t, n, sample_state=invariants)

        t1 = get_log_intermediate_score(x, n, sample_state=invariants)
        t2 = get_log_intermediate_score(x, n - 1, sample_state=invariants)
        t3 = t1 - t2
        logw += t3

        return s2, x, t, logw, invariants

    # Jax functional compatible implementation

    log_weights: Weights = jnp.zeros(n_samples)
    samples: Samples = jnp.zeros(n_samples)
    invariants = get_invariants(n_samples, n_inter)

    sample_init = key, invariants, samples, log_weights
    sample_return = jax.lax.fori_loop(0, n_samples, sample_loop, sample_init)

    key, invariants, samples, log_weights = sample_return

    return samples, log_weights


def get_mean__j(samples: Array = None, weights: Array = None) -> float:

    """params:
     samples: 1d array
     weights: 1d array
    return:
      mean: float"""

    return jnp.sum(samples * weights) / jnp.sum(samples)


def log_neal_interpolating_score_sequence__g(
    x: float = None,
    beta: float = None,
    log_source__j: lPDF = None,
    log_target__j: lPDF = None,
) -> float:

    """As equation (3) from Neal 1998 Annealed Importance Sampling
    Log interpolating distribution
    use partial application of source and target"""

    return beta * log_target__j(x) + (1 - beta) * log_source__j(x)


def nsteps_mh__g(
    key: PRNGKeyArray = None,
    x: float = None,
    log_intermediate__j: lPDF = None,
    intermediate_rv__j: Callable = None,
    n_steps: int = 10,
    kwargs_log_intermediate__j=None,
) -> RV:

    """The transition distribution T(x' | x) implemented using the Metropolis Hastings Algorithm"""

    key, subkey = jax.random.split(key)

    def inner_loop_body(i, val):
        key, x = val
        key, s1, s2 = jax.random.split(key, 3)
        x_prime = x + intermediate_rv__j(s1)  # jax.random.normal(s1)

        # Acceptance prob
        a = log_intermediate__j(
            x_prime, **kwargs_log_intermediate__j
        ) - log_intermediate__j(x, **kwargs_log_intermediate__j)
        a = jnp.exp(a)

        """ 
        if jax.random.uniform(s2) < a:
            x = x_prime
        """
        pred = jnp.array(jax.random.uniform(s2) < a)

        x = jax.lax.cond(
            pred, lambda x, x_prime: x_prime, lambda x, x_prime: x, x, x_prime
        )

        return key, x

    key, x = jax.lax.fori_loop(0, n_steps, inner_loop_body, (key, x))

    return x
