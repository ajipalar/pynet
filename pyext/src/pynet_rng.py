"""
Random functions for pynet built on top of Jax
"""
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.linalg as lin


def positive_definite(key, n):
    """
    Generate a real n-by-n positive definite matrix

    A = LL* where A is a Hermitian matrix and L is a lower triangular matrix
    L* is the conjugate transpose

    """
    ...

def positive_semidefinite(key, n):
    """
    Generate an n-by-n positive semi-definite real matrix

    """
    ...

def chi2(key, k, shape=None, dtype=None):
    """
    Sample a chi2 distributed random variate
    Args:
      k : The parameter of the distribution. The number of degrees of freedom
    Returns
      A random chi2 distributed array with 
    Examples:

    Notes:
      jax.random.gamma a is the shape parameter
      Scaling the variates by 2 sets the scale parameter to two
    """

    return 2 * jax.random.gamma(key, k / 2, shape=shape, dtype=dtype)




def wishart(key, V, p, n):
    """
    Draw a p x p scatter matrix from the Wishart Distribution. O(n^3)
    If V is invertible then S is invertable

    No runtime checks are performed

    Args:
      V : the p x p scale matrix
      p : the rank of the matrix
      n : degrees of freedom with n <= p 
    Returns:
      S : the p x p scatter matrix
    """
    
    keys = jax.random.split(key, )

    c_squared = jax.random




