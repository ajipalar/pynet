import jax
import jax.numpy as jnp
import jax.scipy as jsp

inv = jsp.linalg.inv
def mvn_bart(x, mu, A, L):
    """
    The unormalized log-pdf of the multivariate normal distribution
    

    The Bartlet parameterization of the multivariate normal distributions
    L = cho(cov)
    cov = LAA^TL^T

    """

def mvn_bart_inv(x, mu, A, L):
    """
    The unormalized log-pdf of the multivariate normal distribution using the precision matrix parameterizations 
    """

def wish_bart(A, nu, L, p):
    """
    The unormalized log pdf of the wishart distribution using bartlett decomposition

    Wishart(S, nu, V) -> (A, nu, L)
    V = LL^T
    S = LA(LA)^T
    L = cho(V)
    A diag = c. c^2 ~ chi^2(n-i + 1)
    A off  ~ N(0, 1)
    nu : the scalar degrees of freedom


    L  = cho(V)
    S = LAA^TL^T

    det S = det(L)det(A)det(A)det(L)
    


    Derivation

    inv(V) @ S
    inv(LL^T) @ LA(LA)^T
    inv(L^T)inv(L) @ LA @ (LA)^T
    int(L^T) @ A @ A^T @ L^T
    

    """
    log_det_V = 2 * _log_det_tri(L)
    log_det_S = 2 * _log_det_tri(A) + log_det_V 
    trace_term = - _trace_term(L, A)
    log_gamma = jsp.special.multigammaln(nu , p)
    s = (0.5 * (nu - 1 - p) * log_det_S - (0.5 * trace_term) - (0.5 * (nu * p) * jnp.log(2)) - (0.5 * nu) * log_det_V - log_gamma)  
    return -s 

def wish_bart_from_S(S, nu, L, p):
    log_det_V = 2 * _log_det_tri(L)
    log_det_S = jnp.log(jnp.linalg.det(S)) 
    trace_term = - jnp.trace((inv(L @ L.T) @ S))
    log_gamma = jsp.special.multigammaln(nu , p)

    s = (0.5 * (nu - 1 - p) * log_det_S - (0.5 * trace_term) - (0.5 * (nu * p) * jnp.log(2)) - (0.5 * nu) * log_det_V - log_gamma)  
    return -s 

def _log_det_tri(T):
    return jnp.sum(jnp.log(T.diagonal()))

def _trace_term(L, A):
        return jnp.trace(inv(L.T) @ A @ A.T @ L.T)
