"""
The log_prob module provides access to outer functions
The outer functions build log_prob pure functions that respect jax transformations

log probability mass and density functions
The log base is `e`,(the natural log) unless otherwise noted
Such functions are compatible with jax transforms
"""

import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.stats as stats
from jax.scipy.linalg import inv, det
from jax.scipy.special import multigammaln
from typing import TypeAlias, Callable
from functools import partial

Inner: TypeAlias = Callable

def _cholesky_logdet(scale):
    """
    Following the scipy implementation
    """
    c_decomp = jsp.linalg.cholesky(scale, lower=False)
    logdet = 2 * jnp.sum(jnp.log(c_decomp.diagonal()))
    return c_decomp, logdet


def wishart(p: int) -> Inner:
    """
    Builds the wishart log_prob density function

    Args :
      p the degree of the scatter and scale matrices
    """

    _LOG2 = jnp.log(2)

    def lpdf(X, V, df, /) -> Inner:
        C, log_det_scale = _cholesky_logdet(V) 
        _, log_det_x = _cholesky_logdet(X) # 
        scale_inv_x = jsp.linalg.cho_solve((C, True), X)
        tr_scale_inv_x = scale_inv_x.trace() #jnp.trace(inv(V) @ X)  # could be faster with the cholesky from previous step

        out = ((0.5 * (df - p -1) * log_det_x - 0.5 * tr_scale_inv_x) -
               (0.5 * df * p *_LOG2 + 0.5 * df * log_det_scale + multigammaln(0.5*df, p)))
        return out



    inner_docstring = f"""
        The log density function of the wishart distribution defined over the field of
        real numbers for p = {p}

        Args :
          X the scatter matrix
          V the scale matrix
          df the degrees of freedom. df > p - 1 for p == len(X)

        Returns
          log_prob : float the unormalized log probability density

        Implementation details

        [1] Morris. L Eaton Multivariate Statisitcs A Vector Space Approach

        |A| determinant of A

        No runtime checks are performed so X and V must be positive definite
        """

    lpdf.__doc__ = inner_docstring

    return lpdf 

