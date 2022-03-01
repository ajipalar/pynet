try:
    from IMP.pynet.typedefs import(
        Array, DeviceArray, Dimension, JitFunc, Matrix, PRNGKey, Vector 
    )
    import IMP.pynet.functional_gibbslib as fg
    import IMP.pynet.PlotBioGridStatsLib as nblib
except ModuleNotFoundError:
    from pyext.src.typedefs import(
        Array, DeviceArray, Dimension, JitFunc, Matrix, PRNGKey, Vector 
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
    fn_pdf = jax.scipy.stats.norm.pdf
    
    
    x = np.arange(5, 15, 0.1)
    n_inter = 50
    n_samples = 100
    betas = np.linspace(0, 1, n_inter)
    key = jax.random.PRNGKey(10)
    return mu, sigma, fn_pdf, x, n_inter, n_samples, betas, key
    


def f0_pdf(x, mu, sig):
    """Target distribution: \propto N(mu, sigma)"""
    return np.exp(-(x - mu)**2 / (2 * sig ** 2))

def fn_pdf(x):
    return jax.scipy.stats.norm.pdf

def fn_logpdf(x):
    return jax.scipy.stats.norm.logpdf(x)

def f0_logpdf(x, mu, sig):
    """Log target distribution"""
    return -(x - mu)**2 / (2 * sig ** 2)

def fj_pdf(x, beta, f0_pdf, fn_pdf):
    """Interpolating distribution"""
    return f0_pdf(x)**beta * fn_pdf(x)**(1 - beta)

def fj_logpdf(x, beta, fn_logpdf, f0_logpdf):
    """Log interpolating distribution
       use partial application of f0_pdf, and f_"""
    return beta * f0_logpdf(x) + (1-beta) * fn_logpdf(x)




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

def do_ais(key, n_samples, n_inter, betas, x, f0_pdf):
    """Perform annealed importance sampling as Neal using the N-steps metropolis algoirthm
    """
    samples = jnp.zeros(n_samples)
    weights = jnp.zeros(n_samples)
    fj_pdf = partial(fj_pdf, f0_pdf=f0_pdf)
    for t in range(n_samples):
        # Sample initial point from q(x)
        #x = p_n.rvs() #random variates
        key, subkey = jax.random.split(key)
        x = jax.random.normal(key)
        w = 1

        for n in range(1, len(betas)):
            # Transition
            x = T(subkey, x, lambda x: fj_pdf(x, betas[n]), n_steps=5)

            #Compute weight in log space
            w += jnp.log(fj_pdf(x, betas[n])) - jnp.log(fj_pdf(x, betas[n - 1]))

        samples = samples.at[t].set(x)
        weights = weights.at[t].set(jnp.exp(w))

    return samples, weights

def generic_ais(key,
                ais_prelude : Callable,
                n_samples : Dimension,
                m_interpolating_dist : Dimension,
                f0_pdf,
                fj_pdf,
                f_m,
                T : Callable
                ):
    """Designed for partial application of functions followed by jit compilation
       There are n_samples returned samples and weights.

       There are m_interpolating_dist

       let f_m be the dist of interest
       let fj_pdf be an interpolating dist
           fj_pdf(j, val)

       let T be the markov transition rule

       """
    samples = jnp.zeros(n_samples)
    weights = jnp.zeros(n_samples)
    def inner_loop_body(j : Index, val : tuple):
        subkey, x, fj_pdfargs = val

        #Pass in a NUTS Sampler
        x = T(j, subkey, x, fj_pdf, fj_pdfargs)
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
    fj_pdf = fg.f0  # (xsi, eta1, eta2) -> float 
    pass

    

def get_sqr_ais_weights(key : PRNGKey, 
               n_samples : Dimension, 
               theta : Vector,
               phi : Matrix,
               p : Dimension,
               fj_pdf,
               fn_pdf,
               f0_pdf,
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

         fj_pdf : the function that defines the intermediate distribution
         fn_pdf : the function that defines the starting distribution.
         f0_pdf : the function for the distribution of interest. Note the functions
               fj_pdf, fn_pdf, f0_pdf may or may not be normalized.

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

            # TODO Instead of passing the intermediate distribution fj_pdf as a parameter
            # could jit compile T(key, theta_tilde, phi_tilde)
            # Better would be to vectorize ais and jit compile the entire kernal once
            # therefore call x, w = ais(key)

            x_kj = T(k1, theta_tilde, phi_tilde, fj_pdf) 
        






        

        samples.at[j].set(x)
        weights = weights.at[j].set(jnp.exp(w))





def sampler(key, n_samples : int, 
            n_inter, 
            betas, 
            x,
            f0_pdf : Callable,
            fj_pdf : Callable,
            fn_pdf : Callable,
            T,  # the transition rule
            ) -> tuple[Array, Array]:

    """Annealed Importance Sampling by Neal.
       Obtain expectations or an estimate of the partition function
       of the distribution of interest p0 = 1/z0 * f0_pdf begining
       with the distribution p_n = 1/z_n * fn_pdf using intermediate
       distributions p_j = 1/z_j * fj_pdf
    """


    return samples, weights

#samples, weights = do_ais(key, n_samples, betas, n_inter, x)
#a = 1/np.sum(weights) * np.sum(weights * samples)

def ais_poisson_sqr(n_inter : int,  # the number of intermediate distributions
                    f0_pdf,
                    fn_pdf,
                    fj_pdf,
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
    
    f0_pdf = partial(f0_pdf, mu=mu, sig=sig) 
    fj_pdf = partial(fj_pdf, f0_pdf=f0_pdf, fn_pdf=fn_pdf)

    f0_logpdf = partial(f0_logpdf, mu=mu, sig=sig)
    fj_logpdf = partial(fj_logpdf, f0_logpdf=f0_logpdf, fn_logpdf=fn_logpdf)

    #x = np.arange(-10, 5, 0.1)
    betas = np.linspace(0, 1, n_inter)

    samples = np.zeros(n_samples)
    weights = np.zeros(n_samples)

    for t in range(n_samples):
        x = p_n.rvs()
        w = 1

        for n in range(1, len(betas)):
            x = T(x, lambda x: fj_pdf(x, betas[n]), n_steps=n_gibbs_steps)

            fn = fj_pdf(x, betas[n])
            fnm1 = fj_pdf(x, betas[n-1])
            if fn == 0 or fnm1 == 0:
                print(f"t {t}, n {n}\nx {x}, w {w}\nfn {fn}, fnm1 {fnm1}\nbetas[n] {betas[n]},betas[n-1] {betas[n-1]}")
                return None

            w += np.log(fj_pdf(x, betas[n])) - np.log(fj_pdf(x, betas[n - 1]))

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
              fj_pdf,
              f0_pdf,
              T,
              m_inter : Dimension,
              nsamples : Dimension,
              f0_pdfargs=[],
              f0_pdfkwargs={},
              fj_pdfargs=[],
              fj_pdfkwargs={},
              f_margs=[],
              f_mkwargs={}) -> tuple[DeviceArray]:
    """params:
         f_m : the starting distribution
         fj_pdf : the intermediate distributions
         f0_pdf : the distribution of interest
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
