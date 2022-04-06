import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from functools import partial
from typing import Callable as f
from typing import Protocol, Union
from .typedefs import Dimension, Index, JitFunc, Matrix, Vector, uLogScore, Number
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



def remove_ith_entry__s(a: Union[Vector, Matrix]) -> JitFunc:
    """Specialize the function 'remove_ith_entry' to a vector of length arr_l
    such that it may be jit compiled"""

    # array values
    # array shape
    # array ndim

    ndim = a.ndim
    assert (ndim == 1) or (ndim == 2)
    if ndim == 2:
        assert a.shape[0] == a.shape[1]

    n = len(a)
    nrows = n
    ncols = n

    outshape = n - 1 if ndim == 1 else (nrows, ncols - 1)

    # case 1 i==0 -> remove 0th entry
    start_indices = [1] if ndim == 1 else [0, 1]
    limit_indices = [n] if ndim == 1 else [nrows, ncols]
    # copy entries from [1, n) to [0, n-1)
    f0__j = lambda x: jax.lax.slice(x, start_indices, limit_indices)

    # case 2 0<i<n
    if ndim == 1:

        def fi__j(x, i):
            o = jnp.zeros(outshape)
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
            o = jnp.zeros(outshape)
            o, x = jax.lax.fori_loop(
                0, i, lambda j, t: (t[0].at[:, j].set(t[1][:, j]), x), (o, x)
            )
            o, x = jax.lax.fori_loop(
                i + 1, n, lambda j, t: (t[0].at[:, j - 1].set(t[1][:, j]), x), (o, x)
            )
            return o

    # case 3 i==n
    start_indices = [0] if ndim == 1 else [0, 0]
    limit_indices = [n - 1] if ndim == 1 else [nrows, ncols - 1]
    fn__j = lambda x: jax.lax.slice(x, start_indices, limit_indices)

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
    for i in range(1, int(n)+1):
        logfac += np.log(i)
    return logfac

def get_logfacx_lookuptable(x: Vector):
    lookup = np.zeros(len(x))
    for i, xi in enumerate(x):
        lookup[i] = logfactorial(xi)
    return lookup


def get_eta2__s(theta: Vector, phi: Matrix, x: Vector):

    rm_i__j = remove_ith_entry__s(theta)

    def get_eta2__j(theta: Vector, phi: Matrix, x: Vector, i: Index):
        return theta[i] + 2 * rm_i__j(phi[:, i], i) * jnp.sqrt(rm_i__j(x, i))

    return get_eta2__j

def get_ulog_score__s(theta: Vector, phi: Matrix, x: Vector) -> JitFunc:
    """Generate the unormalized log score jittable kernal for the poisson sqr model"""
    eta2__j = get_eta2__s(theta, phi, x)
    logfacx = get_logfacx_lookuptable(x)
    def get_ulog_score__j(theta: Vector, phi: Matrix, x: Vector, i: Index) -> uLogScore:
        return phi[i, i] * x[i] + eta2__j(theta, phi, x, i) * jnp.sqrt(x[i]) - logfacx[i]

    return get_ulog_score__j


