from pyext.src.typedefs import ( Array, Dimension, 
        DeviceArray, Matrix, JitFunc, PRNGKey, Vector 
)
import pyext.src.functional_gibbslib as fg
import pyext.src.PlotBioGridStatsLib as nblib
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
   """

def ais_prelude():
    mu = 5
    
    sigma = 2
    f_n = jax.scipy.stats.norm.pdf
    
    
    x = np.arange(5, 15, 0.1)
    n_inter = 50
    n_samples = 100
    betas = np.linspace(0, 1, n_inter)
    key = jax.random.PRNGKey(10)
    return mu, sigma, f_n, x, n_inter, n_samples, betas, key
    


def f_0(x, mu, sig):
    """Target distribution: \propto N(mu, sigma)"""
    return np.exp(-(x - mu)**2 / (2 * sig ** 2))

def f_n(x):
    return jax.scipy.stats.norm.pdf

def log_fn(x):
    return jax.scipy.stats.norm.logpdf(x)

def log_f0(x, mu, sig):
    """Log target distribution"""
    return -(x - mu)**2 / (2 * sig ** 2)

def f_j(x, beta, f_0, f_n):
    """Interpolating distribution"""
    return f_0(x)**beta * f_n(x)**(1 - beta)

def log_fj(x, beta, log_fn, log_f0):
    """Log interpolating distribution
       use partial application of f_0, and f_"""
    return beta * log_f0(x) + (1-beta) * log_fn(x)




def T(key, x, f, n_steps=10):
    """Transition distribtuion T(x'|x) using n-steps Metropolis sampler"""
    key, subkey = jax.random.split(key)

    for t in range(n_steps):
        #Proposal
        x_prime = x + jax.random.normal(key)

        #Acceptance prob
        a = f(x_prime) / f(x)
        

        if jax.random.uniform(subkey) < a:
            x = x_prime
    return x

def do_ais(key, n_samples, n_inter, betas, x, f_0):
    """Perform annealed importance sampling as Neal using the N-steps metropolis algoirthm
    """
    samples = jnp.zeros(n_samples)
    weights = jnp.zeros(n_samples)
    f_j = partial(f_j, f_0=f_0)
    for t in range(n_samples):
        # Sample initial point from q(x)
        #x = p_n.rvs() #random variates
        key, subkey = jax.random.split(key)
        x = jax.random.normal(key)
        w = 1

        for n in range(1, len(betas)):
            # Transition
            x = T(subkey, x, lambda x: f_j(x, betas[n]), n_steps=5)

            #Compute weight in log space
            w += jnp.log(f_j(x, betas[n])) - jnp.log(f_j(x, betas[n - 1]))

        samples = samples.at[t].set(x)
        weights = weights.at[t].set(jnp.exp(w))

    return samples, weights

def generic_ais(key,
                ais_prelude : Callable,
                n_samples : Dimension,
                m_interpolating_dist : Dimension,
                f_0,
                f_j,
                f_m,
                T : Callable
                ):
    """Designed for partial application of functions followed by jit compilation
       There are n_samples returned samples and weights.

       There are m_interpolating_dist

       let f_m be the dist of interest
       let f_j be an interpolating dist
           f_j(j, val)

       let T be the markov transition rule

       """
    samples = jnp.zeros(n_samples)
    weights = jnp.zeros(n_samples)
    def inner_loop_body(j : Index, val : tuple):
        subkey, x, f_jargs = val

        #Pass in a NUTS Sampler
        x = T(j, subkey, x, f_j, f_jargs)
        w += None


    def inner_ais_loop(k : Index, val : tuple):
        key = init_val
        key, subkey = jax.random.split(key) 
        x = ais_prelude(subkey) 
        w = 1

        return jax.lax.fori_loop(1, m_interpolating_dist, inner_loop_body, inner_val)


    samples, weights = jax.lax.fori_loop(0, n_samples, inner_ais_loop, init_value)



    x = None


# AIS in the context of sqr models
def get_phi_tilde(phi : Matrix, gamma : float) -> Matrix:
    phi_diag = phi.diagonal()
    phi_tilde = phi * gamma[j]
    phi_tilde = nblib.set_diag(phi_tilde, phi_diag)
    return phi_tilde


def jit_compile_sqr_ais():
    """jit compiles the annealed importance sampling into a single kernal"""
    f_j = fg.f0  # (xsi, eta1, eta2) -> float 
    pass

    

def get_sqr_ais_weights(key : PRNGKey, 
               n_samples : Dimension, 
               theta : Vector,
               phi : Matrix,
               p : Dimension,
               f_j,
               f_n,
               f_0,
               npseed : Vector,
               ngibbs_steps : Dimension,
               T) -> tuple[Array, Array]:
    """Perform Annealed Importance Sampling to estimate the log partition 
       function Anode(eta1, eta2) as Inouye 1998

       params:
         n_samples : the number of weights and samples to obtain
                     where n_samples = N and weight^(i) goes from 1 to N
                     as defined by Neal 1998
         
         theta : the nodewise parameter vector as defined by Inouye 2016.
                 theta \in R^p
         
         phi : the parameter matrix as defined by Inouye 2016. 
               phi_sqr \in R^{p x p}. phi is symmetric, the elements may be 
               positive or negative

         p : The number of nodes in the graph, the dimension of the theta, and
             phi parameter. For proteins in an AP-MS pulldown, the number of 
             proteins in the pulldown

         f_j : the function that defines the intermediate distribution
         f_n : the function that defines the starting distribution.
         f_0 : the function for the distribution of interest. Note the functions
               f_j, f_n, f_0 may or may not be normalized.

         npseed : Vector. A vector defining the np seed for sampling the initial
                          multivariate exponential distribution. 

         
         ngibbs_steps :

       return:
        samples : a vector of the samples         

       """

    samples = jnp.zeros(n_samples)
    weights = jnp.zeros(n_samples)

    gamma = jnp.arange(0, n_samples)
    # scale = beta = 1/-phi_ss for phi_ss == eta1 and eta1 < 0

    #Pr(x | eta1) = Pi s=1, p exp{eta1 -A(eta1)}


    # k index over the final number of samples
    # may be able to vectorize this loop
    for k in range(1, n_samples):
        # sample an initial point

        #Begin at the j=0 condition, sample from the base exponential distribution

        phi_tilde = get_phi_tilde(phi, gamma[0])
        phi_ss_vec = phi_tilde.diagonal() 
        #define x0  # should change the indexing to be reverse as Neal 1998
        x_kj = scipy.stats.expon.rvs(scale= 1/-phi_ss_vec, size=(p), random_state=npseed[k])
        
        # index over the number of intermediate distributions
        for j in range(1, len(gamma)): 
            # Define the intermediate distributions
            phi_tilde = get_phi_tilde(phi, gamma[j])
            theta_tilde = theta * gamma[j]  

            key, k1 = jax.random.split(key)

            # TODO Instead of passing the intermediate distribution f_j as a parameter
            # could jit compile T(key, theta_tilde, phi_tilde)
            # Better would be to vectorize ais and jit compile the entire kernal once
            # therefore call x, w = ais(key)

            x_kj = T(k1, theta_tilde, phi_tilde, f_j) 
        






        

        samples.at[j].set(x)
        weights = weights.at[j].set(jnp.exp(w))





def sampler(key, n_samples : int, 
            n_inter, 
            betas, 
            x,
            f_0 : Callable,
            f_j : Callable,
            f_n : Callable,
            T,  # the transition rule
            ) -> tuple[Array, Array]:

    """Annealed Importance Sampling by Neal.
       Obtain expectations or an estimate of the partition function
       of the distribution of interest p0 = 1/z0 * f_0 begining
       with the distribution p_n = 1/z_n * f_n using intermediate
       distributions p_j = 1/z_j * f_j
    """


    return samples, weights

#samples, weights = do_ais(key, n_samples, betas, n_inter, x)
#a = 1/np.sum(weights) * np.sum(weights * samples)

def ais_poisson_sqr(n_inter : int,  # the number of intermediate distributions
                    f_0,
                    f_n,
                    f_j,
                    phi,
                    phi_diag,
                    theta):

    gamma_arr = jnp.arange(n_inter)  # from 0 to n_inter - 1
    
    def T(x, f, n_steps=10):
        for t in range(n_steps):

            x_prime = x + np.random.randn()

            a = f(x_prime) / f(x)

            if np.random.rand() < a:
                x = x_prime

    for zi in range(n_inter):
        gamma = gamma_arr[zi]
        theta_tilde = theta * gamma
        phi_tilde = phi_off * gamma + phi_diag

        



    return weights, samples

def ais_example(mu, sig, n_samples=600, n_inter=60, n_gibbs_steps=5):
    """The example from Augstinus Kristiadi's blog"""

    def T(x, f, n_steps=10):
        for t in range(n_steps):

            x_prime = x + np.random.randn()

            a = f(x_prime) / f(x)

            if np.random.rand() < a:
                x = x_prime
        return x

    p_n = st.norm(0, 1)
    
    f_0 = partial(f_0, mu=mu, sig=sig) 
    f_j = partial(f_j, f_0=f_0, f_n=f_n)

    log_f0 = partial(log_f0, mu=mu, sig=sig)
    log_fj = partial(log_fj, log_f0=log_f0, log_fn=log_fn)

    #x = np.arange(-10, 5, 0.1)
    betas = np.linspace(0, 1, n_inter)

    samples = np.zeros(n_samples)
    weights = np.zeros(n_samples)

    for t in range(n_samples):
        x = p_n.rvs()
        w = 1

        for n in range(1, len(betas)):
            x = T(x, lambda x: f_j(x, betas[n]), n_steps=n_gibbs_steps)

            fn = f_j(x, betas[n])
            fnm1 = f_j(x, betas[n-1])
            if fn == 0 or fnm1 == 0:
                print(f"t {t}, n {n}\nx {x}, w {w}\nfn {fn}, fnm1 {fnm1}\nbetas[n] {betas[n]},betas[n-1] {betas[n-1]}")
                return None

            w += np.log(f_j(x, betas[n])) - np.log(f_j(x, betas[n - 1]))

        samples[t] = x
        weights[t] = np.exp(w)

    return samples, weights

def gibbs_ais_sqr(key):
    key, k1 = jax.random.split(key)

    samples = jnp.zeros(n_samples)
    weights = jnp.zeros(n_samples)


    # 

    x = jax

    return samples, weights

def gibbs_generic_ais(key,
              f_m,
              f_j,
              f_0,
              T,
              m_inter : Dimension,
              nsamples : Dimension,
              f_0args=[],
              f_0kwargs={},
              f_jargs=[],
              f_jkwargs={},
              f_margs=[],
              f_mkwargs={}) -> tuple[DeviceArray]:
    """params:
         f_m : the starting distribution
         f_j : the intermediate distributions
         f_0 : the distribution of interest
         n_inter : the number of intermediate distributions from [1 to n_inter]
         n_samples : number of samples and weights to return
    """

    # initiate the samples and weights
    samples = jnp.zeros(n_samples)
    weights = jnp.zeros(n_samples)


    # sample the 

    key, k1 = jax.random.split(key, 2)
    x = jax.random.exponential(k1, shape=(nsamples, ))




    return samples, weights
