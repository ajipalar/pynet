import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from functools import partial
from typing import Callable as f
from typing import Protocol, Sequence, Union
from .typedefs import (
    Dimension,
    Index,
    JitFunc,
    Array,
    ArraySquare,
    Array1d,
    uLogScore,
    Number,
    Callable,
    PRNGKeyArray,
)
from . import predicates as pred
import collections

# variable value -> Array index
n_samples: int  # [1, n_samples] -> [0, n_samples)
n_prey: Dimension
n_replicates: int
n_interpolating: int

pynet_jax_fdtype = jnp.float32

# samples, weights = model.sample.ais(n_samples, n_interpolating)

PoissonSQR = collections.namedtuple("PoissonSQR", ["param", "data"])

SQRParam = collections.namedtuple("SQRParam", ["eta1", "eta2", "theta", "Phi"])

AIS = collections.namedtuple("AIS", ["T", "source", "target", "intermediate"])

Source = collections.namedtuple("Source", ["rv"])

Get = collections.namedtuple("Get", ["eta1", "eta2"])


def remove_ith_entry__s(a: Union[Array1d, ArraySquare]) -> JitFunc:
    """Specialize the function 'remove_ith_entry' to a vector of length arr_l
    such that it may be jit compiled"""

    # array values
    # array shape
    # array ndim

    ndim = a.ndim
    assert (ndim == 1) or (ndim == 2)
    if ndim == 2:
        assert pred.is_array_square(a)

    n = len(a)
    nrows = n
    ncols = n

    outshape = n - 1 if ndim == 1 else (nrows, ncols - 1)

    # case 1 i==0 -> remove 0th entry
    start_indices0 = [1] if ndim == 1 else [0, 1]
    limit_indices0 = [n] if ndim == 1 else [nrows, ncols]
    # copy entries from [1, n) to [0, n-1)

    def f0__s(x: Array, s: Sequence[int], l: Sequence[int]):
        return jax.lax.slice(x, s, l)

    f0__j = partial(f0__s, s=start_indices0, l=limit_indices0)
    del f0__s
    del start_indices0
    del limit_indices0

    # case 2 0<i<n
    if ndim == 1:
        assert pred.is_array1d(a)

        def fi__j(x, i):
            o = jnp.zeros(outshape, dtype=x.dtype)
            # copy entries from [0, i) to [0, i)
            o, x = jax.lax.fori_loop(
                0, i, lambda j, t: (t[0].at[j].set(t[1][j]), x), (o, x)
            )
            # copy entries from [i+1, n) to [i, n-1)
            o, x = jax.lax.fori_loop(
                i + 1, n, lambda j, t: (t[0].at[j - 1].set(t[1][j]), x), (o, x)
            )
            return o

    else:

        def fi__j(x, i):
            o = jnp.zeros(outshape, dtype=x.dtype)
            o, x = jax.lax.fori_loop(
                0, i, lambda j, t: (t[0].at[:, j].set(t[1][:, j]), x), (o, x)
            )
            o, x = jax.lax.fori_loop(
                i + 1, n, lambda j, t: (t[0].at[:, j - 1].set(t[1][:, j]), x), (o, x)
            )
            return o

    # case 3 i==n
    start_indicesn = [0] if ndim == 1 else [0, 0]
    limit_indicesn = [n - 1] if ndim == 1 else [nrows, ncols - 1]

    def fn__s(x, s: Sequence[int], l: Sequence[int]):
        return jax.lax.slice(x, s, l)

    fn__j = partial(fn__s, s=start_indicesn, l=limit_indicesn)

    del fn__s
    del start_indicesn
    del limit_indicesn

    # branch2__j = branch2__s(arr_l, zf__s=zf__s, i_eq_arr_l__j=ieqarr__j)

    def branch2__j(a, i):
        o = jax.lax.cond(
            i == n - 1, lambda m, b: fn__j(m), lambda m, b: fi__j(m, b), *(a, i)
        )

        return o

    def remove_ith_entry__j(arr, i):
        out_arr = jax.lax.cond(
            i == 0, lambda a, b: f0__j(a), lambda a, b: branch2__j(a=a, i=b), *(arr, i)
        )
        return out_arr

    return remove_ith_entry__j


def logfactorial(n: Union[int, float]):
    logfac = 0
    for i in range(1, int(n) + 1):
        logfac += np.log(i)
    return logfac


def get_logfacx_lookuptable(x: Array1d):
    lookup = jnp.zeros(len(x))
    for i, xi in enumerate(x):
        lookup = lookup.at[i].set(logfactorial(xi))
    return lookup


def get_eta2__s(theta: Array1d, phi: ArraySquare, x: Array1d):

    rm_i__j = remove_ith_entry__s(theta)

    def get_eta2__j(theta: Array1d, phi: ArraySquare, x: Array1d, i: Index):
        #         sa em                      mm
        return theta[i] + 2 * (rm_i__j(phi[:, i], i) @ jnp.sqrt(rm_i__j(x, i)))

    return get_eta2__j


def get_ulog_score__s(theta: Array1d, phi: ArraySquare, x: Array1d) -> JitFunc:
    """Generate the unormalized log score jit kernal for the poisson sqr model"""
    eta2__j = get_eta2__s(theta, phi, x)
    logfacx = get_logfacx_lookuptable(x)

    def get_ulog_score__j(
        theta: Array1d, phi: ArraySquare, x: Array1d, i: Index, logfacx: Callable
    ) -> uLogScore:
        return (
            phi[i, i] * x[i] + eta2__j(theta, phi, x, i) * jnp.sqrt(x[i]) - logfacx[i]
        )

    get_ulog_score__j = partial(get_ulog_score__j, logfacx=logfacx)

    return get_ulog_score__j


# Functions for AIS sampling of the Poisson SQR Model


# Base Exponential Closed Form Solution
# Not jit, implemented in scipy
# Solution -> precompute and feed in as array


def erf_taylor_approx__s(nmax: int):
    """A jax implementation of the error function erf using the Maclaurin seriess
    see https://wikipedia.org/wiki/Error_function

    the imaginary error function erfi = -i * erf (i * z)
    """

    def erf_taylor_approx__j(z, nmax, facn):
        a = 2.0 / (jnp.sqrt(jnp.pi))
        b = 0.0
        for n in range(0, n):
            b += ((-1) ** n) * (z ** (2 * n + 1)) / (facn[n] * (2 * n + 1))

        return a * b

    logfactable = get_logfacx_lookuptable(nmax)
    facn = jnp.exp(logfactable)
    return erf_taylor_approx__j


def erf_maclaurin_approximation__s(nmax):
    def inner_logsum__s(z, n):
        logb = 0
        for k in range(1, n):
            logb += (2 * jnp.log(-z)) - jnp.log(k)
        return logb

    def mac__s(z, n):
        a = z / (2 * n + 1)
        b = inner_logsum__s(z, n)
        b = jnp.exp(b)
        return a * b

    partial_funcs = []
    for n in range(0, nmax):
        f = partial(mac__s, n=n)
        partial_funcs.append(f)

    def erf_mac_approx__s(z, nmax, partial_funcs):
        a = 2 / jnp.sqrt(jnp.pi)
        b = z

        for n in range(0, nmax):
            b += partial_funcs[n](z)

        return a * b

    erf_mac_approx__j = partial(
        erf_mac_approx__s, nmax=nmax, partial_funcs=partial_funcs
    )
    return erf_mac_approx__j


def erf(z: complex):
    """A jax implementation of the error function based on the 1993 Sun Microsystems
    erf approximation used in s_erf.c. Original Copyright (C)

     @(#)s_erf.c 1.3 95/01/18 */

     ====================================================
     Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.

     Developed at SunSoft, a Sun Microsystems, Inc. business.
     Permission to use, copy, modify, and distribute this
     software is freely granted, provided that this notice
     is preserved.
     ====================================================

    """


"""
def base_exp_z(eta1: complex, eta2: complex) -> float: 

    a = jnp.sqrt(jnp.pi) * eta1
    b = jnp.exp( -eta2**2 / (4 * eta1))
    arg = -eta2 / (2 * jnp.sqrt(-eta1)) # Must be complex! no check
    c = sp.special.erfc(arg) # not jit
    

    def get_z_base_exp(theta, phi, i, get_eta2__j):
        # phi[i, i] != 0
        a = (-1) / (4 * phi[i, i])
        b = (-eta2) / (2 * jnp.sqrt(-phi[i,i]))
        return (
          jnp.sqrt(jnp.pi) * jnp.exp(a) * (1 - erf(b))/(-2 * (jnp.sqrt((-phi[i, i])**3))) - 1/phi[i, i]
        )
"""


def T1__j(
    key: PRNGKeyArray,
    theta: Array1d,
    phi: ArraySquare,
    f: Callable,
    nsteps: int,
    d: Dimension,
    rgen_theta=jax.random.normal,
    rgen_phi=jax.random.normal,
):

    """The n-steps Metropolis Hastings algorithm

    Generic in the scoring function
    TODO: Make the Alogrithm Generic to Arbitrary types

    Args:
      key:
        A jax splittable PRNGKeyArray
      theta:
        A d-length DeviceArray
      phi:
        A (d, d) DeviceArray
      f:
        The probability density function or log probability density function
      nsteps:
        The number of steps the Metropolis Hastings algorithm should take
      d:
        The dimensionality of the problem

    Returns:
      theta:
        A d-length paramter vector
      phi:
        A (d, d) size parameter matrix


    """

    keys = jax.random.split(key, 4)
    for t in range(nsteps):
        theta_prime = theta + rgen_theta(keys[1], (d,))
        phi_prime = phi + rgen_phi(keys[2], (d, d))

        a = f(theta_prime, phi_prime) / f(theta, phi)

        rn = jax.random.uniform(keys[3])

        theta, phi = jax.lax.cond(
            rn < a, (lambda: (theta_prime, phi_prime)), (lambda: (theta, phi))
        )
        keys = jax.random.split(keys[0], 4)

    return theta, phi


def T1_nsteps_mh__s(f: Callable, nsteps: int, d: Dimension, T1__j=T1__j) -> JitFunc:
    # TODO Refactor f to be named scoref
    """Specialize the Transition distribtuion by dimension and scoring function"""
    scoref = f

    T1_nsteps_mh__j = partial(T1__j, f=scoref, nsteps=nsteps, d=d)

    return T1_nsteps_mh__j


def ais__j(
    key: PRNGKeyArray, d: Dimension, nsamples: int, ninterpol: int, T: Callable
) -> tuple[Array]:

    """Annealed Importance Sampling"""

    phi_shape = (d, d)
    theta_shape = (d,)
    x_shape = (d,)

    samples = jnp.zeros(nsamples)
    weights = jnp.zeros(nsamples)

    gammas = jnp.arange(ninterpol)

    keys = jnp.zeros((ninterpol + 1, 2), dtype=jnp.uint32)
    keys = keys.at[0].set(key)

    for i in range(0, nsamples):

        keys = jax.random.split(keys[0], num=ninterpol + 1)

        # Generate an array from a random exponential distribution

        x = jax.random.exponential(
            keys[1], shape=[d]
        )  # Generate a sample from pn, random exp

        for j in range(1, ninterpol):
            # Sample from the transition distribution

            theta, phi = T(keys[j + 1], x)

        samples = samples.at[i].set(x)

    return samples, weights


def ais__s(
    d: Dimension, nsamples: int, ninterpol: int, T: Callable, scoref: Callable, ais__j=ais__j
) -> JitFunc:

    """The partial specializer for the ais__j function"""

    ais__j = partial(ais__j, d=d, nsamples=nsamples, ninterpol=ninterpol, T=T)
    return ais__j
