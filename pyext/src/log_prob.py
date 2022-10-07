"""
The log_prob module provides access to outer functions
The outer functions build log_prob pure functions that respect jax transformations

log probability mass and density functions
The log base is `e`,(the natural log) unless otherwise noted
Such functions are compatible with jax transforms
"""

import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy.linalg import inv, det
from jax.scipy.special import multigammaln
from typing import TypeAlias, Callable
from functools import partial

Inner: TypeAlias = Callable


def norm(x, my, sigma):
    return jnp.exp


def wishart(p: int) -> Inner:
    """
    Builds the wishart log_prob density function

    Args :
      p the degree of the scatter and scale matrices
    """

    def lpdf(X, V, df) -> Inner:
        a = ((df - p - 1) / 2) * jnp.log(det(X))
        b = -jnp.trace(jnp.matmul(inv(V), X)) / 2
        c = (df * p / 2) * jnp.log(2)
        d = (df / 2) * jnp.log(det(V))
        e = multigammaln(df / 2, p)

        return a + b - c - d - e
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
