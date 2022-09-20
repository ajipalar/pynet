"""
Random functions for pynet built on top of Jax
"""
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.linalg as lin


def is_positive_definite(key, n):
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

def standard_wishart(key, p, n):
    """
    
    Compute A drawn from a standard wishart. Analagous to standard normal.
    n > p-1

    A is a matrix whose diagonal are sqrt(chi2)
    Top diagonal are zeros
    Off lower diagonal are normal

    Args:
      key : PRNGKey
          Number of variates to generate
      p : The dimension of the scatter matrix
      n : The degrees of freedom

    Retruns:

    Source:
    """
    # Normal Elements for off diagonal

    keys = jax.random.split(key, num = 2)
    n_tril = p * (p-1) // 2
    covariances = jax.random.normal(keys[0], shape=(n_tril,)) 

    # sqrt(chi-square) of n-i + 1 for diagonal elements

    ks = n - jnp.arange(1, p+1) + 1
    variances = jnp.sqrt(chi2(keys[1], ks))

    # Create an A matrix
    A = jnp.zeros((p, p))

    # Input the covariances
    
    tril_idx = jnp.tril_indices(p, k=-1) # diagonal offset k
    A = A.at[tril_idx].set(covariances)

    # Input the variances

    diag_idx = jnp.diag_indices(p)
    A = A.at[diag_idx].set(variances)

    return A


def _wishart(key, C, n, p):
    """
    Draw random samples from a Wishart distribution

    Args:
      V : the p x p scale matrix
      p : the rank of the matrix
      n  : The degrees of freedom of the matrix
    Returns:
      S : the p x p scatter matrix
    """

    A = standard_wishart(key, p, n)  
    CA = jnp.dot(C, A)
    A = jnp.dot(CA, CA.T)
    return A

def wishart(key, V, n, p):
    """
    Draw random samples from Wishart Distribution
    """
    C = jsp.linalg.cholesky(V, lower=True)
    S = _wishart(key, C, n, p)
    return S



    

