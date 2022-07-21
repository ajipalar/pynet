"""
This module provides the scoring and sampling of d-dimensional poisson
square root graphical models (Inouye 2016) implemented in Jax.


  fname__j are jittable functions
      fname = jax.jit(fname__j)(args=args, kwargs=kwargs)

  fname__g are generic functions that require specialization
      fname__j = partial(fname__g, g_arg_1 = lower_func_1)
      fname = jax.jit(fname__j)(args=args, kwargs=kwargs)

  fname__s are functions that specialize generic functions to yield jittable functions
      fname__j = fname__s(args, kwargs)

Example Usage:


    # Problem Dimension

    d: int = 256  # number of nodes
    rseed: int = 13
    nkeys: int = 4
    keys = jax.random.PRNGKey(rseed, nkeys)
    replicates: int = 3

    # Input information - synthetic data

    X = get_X(d, replicates)  # 256 x 3  

    # Representation

    theta: jnp.array = get_theta(d)
    phi: jnp.array = get_phi(d)
    
    # Scoring Function

    scoref = get_ulog_score

    # Optimization and Monte Carlo Sampling

    nsamples = int(1e6)
    n_gibbs_steps
      ...
    

Dev notes:

    In general functions should be called by keyword. It is easy to add and remove
    keyword arguments from both caller and callee, opposed to positional arguments.
    Additionally funciton calls become more readable.

TODO:
    Implement Function Types (A, B) -> R. Callable[[P], R]

"""
import jax
import jax.random
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from functools import partial
from typing import Any, Callable as f
from typing import Protocol, Sequence, Union
from .typedefs import (
    DeviceArray,
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
    such that it may be jit compiled

    a:
      shape parameter. TODO refactor to shape


    """

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
            o = jnp.zeros(shape=outshape, dtype=x.dtype)
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
    lookup = jnp.zeros(shape=len(x))
    for i, xi in enumerate(x):
        lookup = lookup.at[i].set(logfactorial(xi))
    return lookup


def get_eta2__s(theta: Array1d, phi: ArraySquare, x: Array1d):

    rm_i__j = remove_ith_entry__s(a=theta)

    def get_eta2__j(theta: Array1d, phi: ArraySquare, x: Array1d, i: Index):
        #         sa em                      mm
        return theta[i] + 2 * (rm_i__j(arr=phi[:, i], i=i) @ jnp.sqrt(rm_i__j(arr=x, i=i)))

    return get_eta2__j


def get_exponent__s(theta: Array1d, phi: ArraySquare, x: Array1d) -> JitFunc:
    """Generate the unormalized log score jit kernal for the poisson sqr model"""
    eta2__j = get_eta2__s(theta=theta, phi=phi, x=x)
    logfacx = get_logfacx_lookuptable(x=x)

    def get_exponent__j(
        theta: Array1d, phi: ArraySquare, x: Array1d, i: Index, logfacx: Callable
    ) -> uLogScore:
        """Returns the log unormalized score for poisson sqr
           
           phi[i,i] * x[i] + eta2 * sqrt(x[i]) - log x[i]!

           params:
             theta: d dimensional real array
             phi: d x d dimensional real array
             x: d dimensional non-negative integer array
             i: float, index
           returns:
             u_log_score: float
        """
        return (
            phi[i, i] * x[i] + eta2__j(theta=theta, phi=phi, x=x, i=i) * jnp.sqrt(x[i]) - logfacx[i]
        )

    get_exponent__j = partial(get_exponent__j, logfacx=logfacx)

    return get_exponent__j


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
      rgen_theta : (K, SHAPE) -> 

    Returns:
      theta:
        A d-length paramter vector
      phi:
        A (d, d) size parameter matrix


    """

    """ 
    for t in range(nsteps):
        theta_prime = theta + rgen_theta(keys[1], (d,))
        phi_prime = phi + rgen_phi(keys[2], (d, d))

        a = f(theta_prime, phi_prime) / f(theta, phi)

        rn = jax.random.uniform(keys[3])

        theta, phi = jax.lax.cond(
            rn < a, (lambda: (theta_prime, phi_prime)), (lambda: (theta, phi))
        )
        keys = jax.random.split(keys[0], 4)
    """

    def fori_loop_body(loop_index, init_val):
        theta, phi, keys = init_val

        theta_prime = theta + rgen_theta(key=keys[1], shape=(d,))
        phi_prime = phi + rgen_phi(key=keys[2], shape=(d, d))

        a = f(theta_prime, phi_prime) / f(theta, phi)

        rn = jax.random.uniform(keys[3])

        theta, phi = jax.lax.cond(
            rn < a, (lambda: (theta_prime, phi_prime)), (lambda: (theta, phi))
        )
        keys = jax.random.split(keys[0], 4)
        return theta, phi, keys

    keys = jax.random.split(key, 4)

    init_val = theta, phi, keys

    theta, phi, keys = jax.lax.fori_loop(0, nsteps, fori_loop_body, init_val)

    return theta, phi


def T1_nsteps_mh__s(f: Callable, nsteps: int, d: Dimension, T1__j=T1__j) -> JitFunc:
    # TODO Refactor f to be named scoref
    """Specialize the Transition distribtuion by dimension and scoring function"""
    scoref = f

    T1_nsteps_mh__j = partial(T1__j, f=scoref, nsteps=nsteps, d=d)

    return T1_nsteps_mh__j

def T_gibbs__s(theta, phi, x, n_gibbs_steps, get_eta2__j) -> Callable[Any, float]:

    """A gibbs sampler for the transition distribtuion within the AIS sampling algorithm

       params:
         theta: 1dArray 
           A paramter array implementing a vector
         Phi: 2dArray
           A 2d parameter array implementing a square symmetric matrix
         x: 1dArray
           A Spectral Counts data array implementing a vector


        By equation 4) in Square Root Graphical Models:  Multivariate Generalizations of
        Univariate Exponential Families that Permit Positive Dependencies by Inouye et al

        4) A(theta, phi) = integral exp { thetaT * sqrt(x) + sqrt(x)T * Phi * sqrt(x) + sum(B(x)) }

        Dev Notes on Gibbs sampling from Fixed-Length Poisson MRF:
        Adding Dependencies to the Multinomia

          p: number of words
          n: number of documents
          k: number of topics

        Poisson Markov Random Field:
          theta: node vector
          phi:   edge matrix

          Pr_PMRF(x|theta, phi) = exp {thetaT * x + xT * Phi * x - sum(s=1,p)(log(xs!)) - A(theta, phi) }

          The conditional distribtuion of 1 word ("One Spectral counts") given all other spectral counts is a 1d Poisson distribution

          Pr(xs | x_-s) with a natural parameter eta_s = theta_s + xT_(-s) * Phi_s

          1d Poisson in natural form is
            eta = log(lambda)
            lambda = exp {eta}
            lambda = exp {theta_s + xT_(-s) * Phi_s}
            Poiss(x| eta) = exp {eta * x - log(x!) - exp(eta))


    1. Generate Spectral Counts Data From Independant Base Exponential
       Distribution

    """

    def gibbs_loop_body__s(key, y, theta, phi, get_eta2__j: Callable, d) -> Callable:

        def body(s, params) -> tuple:
            keys, y, theta, phi, spec_counts_array  = params
            natural_rate = get_eta2__j(theta, phi, y, s)
            lam = jnp.exp(nartual_rate)
            xs = jax.random.poisson(keys[s], lam)
            spec_counts_array = spec_counts_array.at[s].set(xs)

            return key, y, theta, phi, spec_counts_array

        def gibbs__j(key, y, theta, phi, get_eta2__j, d, body) -> Array1d:

            keys = jax.random.split(key, num=d)
            spec_counts_array = jnp.zeros(d)
            params = keys, y, theta, phi, spec_counts_array
            params = jax.lax.fori_loop(0, d, body, params)

            return spec_counts_array

        init_params = ...

        gibbs_loop_body__j = partial(gibbs_loop_body__s, get_eta2__j=get_eta2__j, d=d, body=body)

        T_gibss__j = jax.lax.fori_loop(0, n_gibbs_steps, gibbs_loop_body__j, init_params)

        return gibbs_loop_body__j


    gibbs_loop_body__j = gibbs_loop_body__s(key, y, theta, phi, get_eta2__j, d, body)
    return gibbs_loop_body__j


def T__j(key, theta, phi, x_prime, n_gibbs_steps) -> float:
    """Generate a sample x ~ T(x | x')"""

    # Do Gibbs Sampling n_steps times
      
      # Slice sample
    
      # Slice sample



    # Evaluate the score function f(theta, phi, x)
    

    return x
    


#### Helper Functions for the Gibbs sampling Alagorithm of the Poisson SQR Model ###

def get_poisson_sqr_node_conditional_rv__s(key, theta, phi, x, i, get_eta2__j):
    """The node conditional distributions are proportional to
       the univariate poisson distribution with respect to eta1 and eta2

       Meaning that if we hold theta, phi, and all x_minus s not xs constant

       then xs is distrbuted according to some exponential family distribution (poisson or base exponential) and can be sampled
       Performing this sequentially should yield a gibbs sampler.

       Specifically begin

       (x0, x1, x2, x3, ..., xN) = x

       x0' ~ p(x0 | theta, phi, x1...xN)
       x1' ~ p(x1 | theta, phi, x0', x2,...xN)
       ...
       xN' ~ p(xN | theta, phi, x0',..., xN-1')

       x = (x0', x1', x2', x3', ..., xN')
       repeat for N gibbs steps

       
       params:
         key:
           A jax.PRNGKeyArray
         theta:
           Parameter vector of length d
         phi:
           Parameter matrix of size d x d
         x:
           A data vector of length d
         i:
          an index from [0, d)
         get_eta2__j:
           a jittable function whose signature is (theta, phi, x, i) -> float


       Proof

       Pr(x| lambda) = lambda^x exp{-lambda) / x!
                     = exp{ log(lambda ^x) - log(x!) -lambda}
                     = exp{ x*log(lambda) + (- log(x!)) -lambda}
                     = exp{ eta*x + B(x) -lambda}

       Pr(x|eta) = exp(eta * x -log(x!) -exp(eta))
         eta = log(lambda)
         B(x) = -log(x!)
         A(eta) = exp(eta)

       
       The Node Conditional Distribution for the Poisson Square Root Graphical Model
       is given by equation 5)
       
         xs: x at s e.g., x[s]
         x_s: x minus s

       5) Pr(xs| x_s, theta, phi) = exp(eta1*sqrt(xs) + eta2*sqrt(xs) + B(xs) - Anode(eta))

       Or equivalently by equation 4)

       If eta2 == 0 then the node conditionals are the base exponential family distribution

    """

    eta2 = get_eta2__j(theta, phi, x)

    # Node Conditional Distribution

    # p(xs|x_s, theta, phi) \propto exp{a -log(xs!)}
    # a = phi_ii * xi + (theta_i + 2*phi_i,-i * sqrt(x_-i)



def ais__j(
    key: PRNGKeyArray, d: Dimension, nsamples: int, ninterpol: int, T: Callable
) -> tuple[Array]:
    

    """Annealed Importance Sampling from Radford M Neal 1997
       Indicies as defined in Neal 1997

       x: theta, phi, y
       points 1...N where N=nsamples
       i ranges from 1...N
       
       w(i) an importance weight = f(x(i))/g(x(i))

       0<j<n where pn = starting distribution
       p0 = final distribtuion

       pn is the Base independant exponential distribution
       p0 is the Full Poisson SQR distribtuion

       A0 = log Z0

       A0 = log 1/N Sum(weights) * Zn

       Zn -> Evaluate and implement using scipy (not jax yet)
         for complex valued functions

       
       Fixing theta and phi

         1. Generate a Spectral counts sample from Base Exp yn-1

         2. Have yn-1, theta, phi 

         3. Tn-1 - assign theta, phi, yn-1
            
            3a. Assign the sequence yn-2 according to n_gibbs_steps

        Parameters:
          T:
            A callable transition distribution with the signature T(key, x)
            What is x? T(key, x) -> x. The scoring function is encoded in the distribution T
                

    """

    phi_shape = (d, d)
    theta_shape = (d,)
    x_shape = (d,)

    samples = jnp.zeros(nsamples)
    weights = jnp.zeros(nsamples)

    gammas = jnp.arange(ninterpol)

    keys = jnp.zeros((ninterpol + 1, 2), dtype=jnp.uint32)
    keys = keys.at[0].set(key)


    def set_phi_diag(i, val):
        phi, phi_tilde = val

        phi_tilde = phi_tilde.at[i, i].set(phi[i, i])

        phi, phi_tilde = val

        return val


    for i in range(0, nsamples):

        keys = jax.random.split(keys[0], num=ninterpol + 1)

        # Generate an array from a random exponential distribution

        x = jax.random.exponential(
            keys[1], shape=[d]
        )  # Generate a sample from pn, random exp

        for j in range(1, ninterpol):
            # Sample from the transition distribution

            # Set the values for theta~ and phi~ with invariant theta and phi
            theta_tilde = theta * gammas[j]
            phi_tilde = gammas[j] * phi
            phi, phi_tilde = jax.lax.fori_loop(0, d, set_phi_diag, (phi, phi_tilde))
            ###

            # The transition probability is a function of theta_tilde and phi_tilde and x
            # It is a random number generator

            x = T(keys[j + 1], theta_tilde, phi_tilde, x)
            
            # x' ~ T(x'| x)
            # MH scoref
            # gamma * theta, phi_off * gamma + phi_diag
            # scoref(theta, phi * gamma, x)

        samples = samples.at[i].set(x)

    return samples, weights


def ais__s(
    d: Dimension,
    nsamples: int,
    ninterpol: int,
    T: Callable,
    scoref: Callable,
    ais__j=ais__j,
) -> JitFunc:

    """The partial specializer for the ais__j function"""

    ais__j = partial(ais__j, d=d, nsamples=nsamples, ninterpol=ninterpol, T=T)
    return ais__j


def plot_surface(x, y, z, import_dependencies=False):

    if import_dependencies:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator
        import numpy as np

    fig, ax = plt.subplots(subplot_kw={'projection': "3d"})

    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))

    ax.zaxis.set_major_formatter('{x:.02f}')

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def slice_sweep__s(key, 
        x: float, 
        pstar: Callable, 
        w: float, 
        pstar_args: tuple = (), 
        pstar_kwargs: dict = {},
        uppstar_args=(lambda pt: ()),
        uppstar_kwargs=(lambda pt: {})
        ) -> tuple:

    """

    TODO: refactor to slice_sweep__g naming convention

    Univariate Slice Sampling

    (x, u) -> (x', u')
    (x, p*(x) -> (x', p*(x'))


    In: PyTree

    Out: (key, x, x_prime, xl, xr, u_prime, t, loop_break)

    Params:
    
      key: 
        A jax.random.PRNGKeyArray
      x: 
        A starting coordinate for the sweep within the domain of pstar
      pstar: 
        A univariate (1-dimensional) probability mass or density function of one parameter.
        p(x)=1/Z*pstar(x). Thus pstar does not have to be normalized

        Siganture constraint

      
      
        pstar args and kwargs are specified so that pstar may be entirely generic.
        These pytrees may change value but not shape
        slice_sweep__s is jittable after specialization on pstar
      
      
      pstar_args[optional]: tuple (jax pytree)
        The *args to pstar that consist of fixed size arrays. Immutable.

      pstar_kwargs[optional]: dict (jax pytree)
        The **kwargs to pstar. Immutable. 

      w: A
        weight parameter for the stepping our algorithm in step 3.
      
    Returns:
      x_prime, u_prime
                  _______          _______

    Invariants

    Constraints:
      pstar(x: float, *pstar_args, **pstar_kwargs)
    
    Maps a point x, u under the density function pstar to x' u'

    This function is meant to be jit compiled to XLA using Jax.
    The default update functions uppstar_args and uppstar_kwargs
    return empty args and kwargs containers.
    Therefore they do not effect the jaxpr representation. 
    
    Folowing David MacKay Book

    Advantages:

    No need for tuning (opposed to Metropolis). src wikipedia
    Automatically adjusts the step size to match the local shape
    of the density function.
    Easier to implement than gibbs.
  
    Random variates exhibit seriel statistical dependance.
  
    For P(x) = 1/Z * P*(x)
    Thus P*(x) \propto P(x)
  
    MacKay Pseudocode
  
    1. evaluate P*(x)
    2. draw a vertical coordinate u' ~ Uniform(0, P*(x))
    3. create a horizontal interval (xl, xr) enclosing x
    4. loop {
    5.   draw x' ~ Uniform(xl, xr)
    6.   evaluate P*(x')
    7.   if P*(x') > u' break out of loop 4-9
    8.   else modify the interval (xl, xr)
    }
    
    """
    
    k1, k2, k3, k4 = jax.random.split(key, 4)
    # step 1 evaluate pstar(x)
    u = pstar(x, *pstar_args, **pstar_kwargs)
    
    # step 2 draw a vertical coordinate
    u_prime = jax.random.uniform(k1, minval=0, maxval=u)
    
    # step 3 create a horizontal interval (xl, xr) enclosing x
    r = jax.random.uniform(k2, minval=0, maxval=1)
    
    xl = x - r * w
    xr = x + (1 - r) * w
    
    xl = jax.lax.while_loop(lambda xl: pstar(xl) > u_prime, lambda xl: xl - w, xl)
    xr = jax.lax.while_loop(lambda xr: pstar(xr) > u_prime, lambda xr: xr + w, xr)
    
    # step 4 loop 1st iteration
    
    loop_break = False
    
    # step 5 draw x'
    x_prime = jax.random.uniform(k3, minval=xl, maxval=xr)

    #Optional update to pstar args and kwargs

    pstar_args = uppstar_args((x_prime, pstar_args, pstar_kwargs))
    pstar_kwargs = uppstar_kwargs((x_prime, pstar_args, pstar_kwargs))
    
    # step 6 evaluate pstar(x')
    t = pstar(x_prime, *pstar_args, **pstar_kwargs)
    
    
    def step7_true_func(val):
        """Do nothing break out of loop"""
        key, x, x_prime, xl, xr, u_prime, t, loop_break = val
        loop_break = True
        return key, x, x_prime, xl, xr, u_prime, t, loop_break

    def step8(val):
        """Perform the shrinking method for step 8"""
        key, x, x_prime, xl, xr, u_prime, t, loop_break = val       
        
        x_prime, x, xr, xl = jax.lax.cond(
            x_prime > x, 
            lambda x_prime, x, xr, xl: (x_prime, x, x_prime, xl),  # reduce the right side
            lambda x_prime, x, xr, xl: (x_prime, x, xr, x_prime),  # reduce the left side
            *(x_prime, x, xr, xl))
        
        return key, x, x_prime, xl, xr, u_prime, t, loop_break
    
    def step7_and_8(val):
        val = jax.lax.cond(
            val[6] > val[5], # p*(x')>u'
            step7_true_func, # do nothing. Break out of loop
            step8, # step 8 modify the interval (xl, xr)
            val)
        
        return val

    # step 7 if pstar(x') > u' break out of loop. else modify interval
    
    val = k4, x, x_prime, xl, xr, u_prime, t, loop_break
    val = step7_and_8(val)

    def step4_loop_body(val):
        
        # step 5 draw x'
        key, x, x_prime, xl, xr, u_prime, t, loop_break = val 
        key, subkey = jax.random.split(key)
        x_prime = jax.random.uniform(subkey, minval=xl, maxval=xr)
        
        # step 6 evaluate pstar(x')
        t = pstar(x_prime)
        
        # step 7
        
        val = key, x, x_prime, xl, xr, u_prime, t, loop_break
        val = step7_and_8(val)
        return val
    
    # End 1st loop iteration
    # Continue the loop executing the while loop
    
    def while_cond_func(val):
        """Check the loop break condition,
           terminate the loop if True"""
        key, x, x_prime, xl, xr, u_prime, t, loop_break = val
        return loop_break == False
    
    val = jax.lax.while_loop(
        while_cond_func, # check the loop break condition
        step4_loop_body, 
        val) # u_prime <= p*(x') i.e., t
        
    return val

def gibbs__step__s(key, theta, phi, xarr, w):
    
    # Make PRNGKeys
    
    k0, k1 = jax.random.split(key)
    
    # Generate from node conditional 0
    
    pstar0 = partial(pstar, i=0, theta=theta, phi=phi, xarr=xarr)
    val = slice_sweep__s(key=k0, x=xarr[0], pstar=pstar0, w=w)
    old_key0, x0, x0_prime, xl, xr, u_prime, t, loop_break = val
    
    # Update the joint independant variable
    xarr = xarr.at[0].set(x0_prime)
    
    # Generate from node conditional 1
    pstar1 = partial(pstar, i=1, theta=theta, phi=phi, xarr=xarr)
    val = slice_sweep__s(key=k2, x=xarr[1], pstar=pstar1, w=w)
    
    
    val = slice_sweep__s(key=k1, x=x)
    old_key1, x1, x1_prime, xl, xr, u_prime, t, loop_break = val
    
    return (x0_prime, x1_prime)


def gibbs_step_d_dimensions__s(key, d, theta, phi, xarr):
    """Take a gibbs step for a d-dimensional poisson sqr model
       generate the spectral counts vector xarr for given values of theta and phi"""

    key_array = jax.random.split(key, d)

    get_ulog_target__j = get_exponent__s(theta, phi, xarr)
    

    def body(i, val):
        
        key_array, theta, phi, xarr = val


    # Sample node conditional 0

    # Sample to node conditional N

    ...

            
def gibbs_step(key, xarr): 
    
    """Defines a single gibbs step for a bi-variate distribution.
       
       (K, T, F, F) -> T
       
       params:
         key:
         xarr:
           A (2, ) dimensional DeviceArray
         rv_cond0:
           (key, Number) -> (key, Number, Number, ... )
           A function to generate random variates from the first conditional distribution
         rv_cond1:
           (key, Number) -> (key, Number, Number, ... )
           A function to generate random variates from the second conditional distribution

       return:   
         xarr:
       
       """
    
    k1, k2 = jax.random.split(key)
    
    # Sample node conditional 0
    #val = slice_sweep(k1, x=xarr[0], pstar=npstar0, w=w)
    
    val = rv_cond0(k1, x=xarr[0])
    old_key0, x0, x0_prime, xl, xr, u_prime, t, loop_break = val
    
    # Update slice
    xarr = xarr.at[0].set(x0_prime)
    # Sample Node conditional 1
    
    #val = slice_sweep(k2, x=xarr[1], pstar=npstar1, w=w)
    val = rv_cond1(k2, x=xarr[1])
    old_key1, x1, x1_prime, xl, xr, u_prime, t, loop_break = val
    xarr = xarr.at[1].set(x1_prime)
    
    return xarr

def gibbs_sampler(key, n_steps: int, x_init, gibbs_step, d=2) -> DeviceArray:
        
    """
    The Gibbs sampling algorithm for a bivariate score

    (K, int, T, F) -> [T] 

    Params:
      key: K
        A jax.random.PRNGKey
      n_steps: int
        The number of Gibbs steps to take
      x_init: T
        The initial value of some type T      
      gibbs_steps: Callable
        (K, T) -> T


    for step in range(n_steps):
        print(f'Sampling step {step}')
        x_init = gibbs_step(keys[step], x_init)
        samples = samples.at[step].set(x_init)
    """
    
    keys = jax.random.split(key, n_steps)
    samples = jnp.zeros((n_steps, d))

    def gibbs_body_fun(i, val):
        keys, x_init, samples = val
        x_init = gibbs_step(keys[i], x_init)
        samples = samples.at[i].set(x_init)
        val = keys, x_init, samples
        return val
    
    val = keys, x_init, samples
    val = jax.lax.fori_loop(0, n_steps, gibbs_body_fun, val)
    keys, x_init, samples = val

    
    return samples


def get_Asqr__s(method,
        ) -> Callable[Any, float]:
    """
    Yields a function that may be used to estimate the value of the normalizing
    constant of the poisson sqr scoring function.

    The yielded jittable function has the signature

    get_Asqr__j(theta, phi, x) -> float

    
    """
    ...


"""

1. Define f(x). Where f is unorm poisson sqr and x is theta, phi, xarr
2. Sample over xarr to normalize using
   - set the sampling hyper parameters
   - get the weights and samples from AIS sampling
   -  
3. Evaluate p(x)

"""
