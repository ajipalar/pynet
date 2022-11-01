"""
Random functions for pynet built on top of Jax
"""
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy.linalg import (
        cholesky,
        inv
)
from jax.random import (
        gamma,
        split,
        normal
)

def chi2(key, k, shape=None, dtype=None):
    """
    Sample a chi2 distributed random variate
    Args:
      k : The parameter of the distribution. The number of degrees of freedom
    Returns
      A random chi2 distributed array with 
    Examples:

    Notes:
      gamma a is the shape parameter
      Scaling the variates by 2 sets the scale parameter to two
    """

    return 2 * gamma(key, k / 2, shape=shape, dtype=dtype)

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

    keys = split(key, num = 2)
    n_tril = p * (p-1) // 2
    covariances = normal(keys[0], shape=(n_tril,)) 

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
    C = cholesky(V, lower=True)
    S = _wishart(key, C, n, p)
    return S

def GWishart():
    """
    Following Alex Lenowski 2013

    K_star in symmetric p x p positive definite matrices
    Data p(D|K) ~ N(0, K-1)
    
    G is a conditional independance graph
    G Pg \sub Pp such that K \in Pg implies that K \in Pp and Kij = 0

    Wg(d, D) pr(K|d, D, G)

    Tcj, A: Pg -> Pg

    1. Set W = Sigma
    2. For j=1, ...,
    
    
    K_star from W(d,D)
    Sig = (Kstar)-1
    K(0) = eye
    


    """

def GWishart(key, K_star, A, J):
    """
    A direct G-Wishart sampler from Lenowitz

    Params:
      key : A jax prng key
      K_star : 
        A positive definite precision matrix distributed according to the Wishart
        distribution
      A : The adjacency matrix representation of the undirected graph



    """

    Init = namedtuple("Init", "W A") 

    def _get_neighbor_indicies(j, A):
        return jnp.where(A[j] != 0)

    def body_fun(j, init):
        """

        Math definitions from Lenkoski
        Cj is a clique in G

        clique : A subset such that every two are adjacent
        neighbor :    

        p is the rank of the matrix
        E sub V x V
        G in Pg
        Nj in V

        beta_j_star in R(p-1)_ (Nj, Nj) (Nj, 1)

        w is 
          p x p
          Wnj is (Nj, Nj)
          Sigma Nj (Nj, j)

          beta_j 

        # c. Replace Wj, -j and W -j, j with W-j, j * beta_j

        """
        col_j = init.A[:, j] # some unknown number of nonzeros elements

        # zero padding

        zp = jnp.zeros
        
        neighbors = get_neighbors(init.A)

        neighbors_j = Wnj
        #Sigma_nj = 

    # Initialization
    init = (Sigma, A) # 1. W = Sigma

    
    W = jax.lax.fori_loop(1, J, body_fun, W)
    K = inv(W) # 4.
    return K
