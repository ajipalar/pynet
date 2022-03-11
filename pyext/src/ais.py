try:
    from IMP.pynet.typedefs import(
        Array, DeviceArray, Dimension, JitFunc, Matrix, Number, PartialF, 
        PDF, lPDF, PMF, lPMF, PRNGKey, PureFunc, Samples, Vector, Weights,
        RV, fParam, iParam, Prob
    )
    import IMP.pynet.functional_gibbslib as fg
    import IMP.pynet.PlotBioGridStatsLib as nblib
except ModuleNotFoundError:
    from pyext.src.typedefs import(
        Array, DeviceArray, Dimension, JitFunc, Matrix, Number, PartialF, 
        PDF, lPDF, PMF, lPMF, PRNGKey, PureFunc, Samples, Vector, Weights,
        RV, fParam, iParam, Prob
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
import distributions as dist

"""This module contains functions that implement Annealed Importance Sampling.
   In particular Annealed Importance Sampling is implemented for the Poisson 
   Square Root Graphical Model as described in Inouye 2016.
   
   In general AIS is implemented using a generic algorithm that is may be jit 
   compiled using jax's jit compiler for use on the XLA kernal.

   The arguments to set_up_ais allow for jit compilation.

   The functions below are implemented as pure functions with no side effects.
   """

# dist.norm.pdf
# dist.norm.lpdf
# dist.norm.rv



def ais_prelude() -> tuple:
    mu = 5
    
    sigma = 2
    fn_pdf = jax.scipy.stats.norm.pdf
    
    
    x = np.arange(5, 15, 0.1)
    n_inter = 50
    n_samples = 100
    betas = np.linspace(0, 1, n_inter)
    key = jax.random.PRNGKey(10)
    return mu, sigma, fn_pdf, x, n_inter, n_samples, betas, key
    
def fj_pdf__g(x : float  = None, 
           beta : float  = None, 
           target__j : PDF = None, 
           source__j : PDF = None):
    """A univariate annealing interpolating distribution between
       the begining distribution f_n and the target f_0
       Designed for target = f0_pdf and source = fn_pdf
       """

    return target__j(x)**beta * source__j(x)**(1 - beta)

def fj_logpdf__g(x : float = None, 
              beta : float = None, 
              log_source__j : lPDF = None, 
              log_target__j : lPDF = None):
    """Log interpolating distribution
       use partial application of f0_pdf, and f_"""
    return beta * log_target__j(x) + (1-beta) * log_source__j(x)


def T_nsteps_mh__py(key, 
                    x: Number = None, 
                    pdf : PDF = None, 
                    n_steps=10) -> float:
    """Transition distribtuion T(x'|x) using n-steps Metropolis sampler"""

    for t in range(n_steps):
        #Proposal
        key, s1, s2 = jax.random.split(key)
        x_prime = x + jax.random.normal(s1)

        #Acceptance prob
        a = pdf(x_prime) / pdf(x)
        

        if jax.random.uniform(s2) < a:
            x = x_prime
    return x

def T_nsteps_mh__g(key : PRNGKey = None, 
         x : Number = None, 
         intermediate__j : PDF = None, 
         intermediate_rvs__j : Callable = None,
         n_steps : int = 10,
         kwargs_intermediate__j = None) -> RV:
    """The transition distribution T(x' | x) implemented using the Metropolis Hastings Algorithm"""

    key, subkey = jax.random.split(key)

    def inner_loop_body(i, val):
        key, x = val
        key, s1, s2 = jax.random.split(key, 3)
        x_prime = x + intermediate_rvs__j(s1) #jax.random.normal(s1)

        #Acceptance prob
        a = intermediate__j(x_prime, **kwargs_intermediate__j) / intermediate__j(x, **kwargs_intermediate__j)
        
        if jax.random.uniform(s2) < a:
            x = x_prime

        return key, x

    key, x = jax.lax.fori_loop(0, n_steps, inner_loop_body, (key, x))

    return x

def T_nsteps_mh__unorm2unorm__p(mu : fParam, sig : fParam) -> JitFunc:
    source__j = fn_pdf__j

    target__j = f0_pdf__j
    target__j : Callable[[RV], Prob] = partial(target__j, mu=mu, sig=sig)

    intermediate_rvs__j = jax.random.normal

    intermediate__j : Callable[[RV, fParam], Prob]
    intermediate__j  = partial(fj_pdf__g, source__j = source__j, target__j = target__j)

    T_nsteps_mh__unorm2unorm__j : Callable[[PRNGKey, RV, dict], RV]

    T_nsteps_mh__unorm2unorm__j = partial(T_nsteps_mh__g,
            intermediate_rvs__j = intermediate_rvs__j,
            intermediate__j = intermediate__j)

    return T_nsteps_mh__unorm2unorm__j

def do_ais__g(key : PRNGKey = None, 
           n_samples : Dimension = None,
           n_inter : Dimension = None, 
           betas :  Array = None, 
           target__j: PDF = None, 
           intermediate__j: PDF = None, 
           source__j: PDF = None,
           source_rvs__j: Callable = None,
           transition_rule__j : Callable = None,
           ) -> tuple[Samples, Weights]:
    """Perform the following steps
       1) partial function application
       2) AIS
    """
    samples : Samples = jnp.zeros(n_samples)
    weights : Weights = jnp.zeros(n_samples)
    
    for t in range(n_samples):
        # Sample initial point from q(x)
        #x = p_n.rvs() #random variates
        key, subkey = jax.random.split(key)
        x = source_rvs__j(key) #jax.random.normal(key)
        logw = 1

        for n in range(1, n_inter):
            # Transition
            #x = transition_rule__j(subkey, x, lambda x: intermediate_j(x, betas[n]), n_steps=5)

            x = transition_rule__j(subkey, x, *args_transition_rule__j, **kwargs_transition_rule__j)

            #Compute weight in log space
            w += jnp.log(intermediate__j(x, betas[n])) - jnp.log(intermediate__j(x, betas[n - 1]))

        samples = samples.at[t].set(x)
        weights = weights.at[t].set(jnp.exp(w))

    return samples, weights

   # convention
   # fname__context_name__tag
   # tags
   # __p : partial application
   # __j : jittable function
   # __g : generic function
   # fname_context_name__p(fname__g, *args, **kwargs) -> fname__context_name__j
   # __py : py function. Not jittable

def do_ais__unorm2unorm__p(mu : float = None,
                        sig : float = None,
                        n_samples : Dimension = None,
                        n_inter : Dimension = None,
                        n_mh_steps : int = None) -> JitFunc:
    """Apply the normal context to the do_ais__g generic sampler"""
    
    target__j : Callable[[RV], Prob] = partial(f0_pdf__j, mu=mu, sig=sig)
    source__j : Callable[[RV], Prob] = fn_pdf__j 
    interm__j : Callable[[RV, fParam], Prob] = partial(fj_pdf__g,
            target__j = target__j,
            source__j = source__j)

    trans_rule__j : Callable[[PRNGKey, RV], RV]
    trans_rule__j = partial(T_nsteps_mh__g, 
            intermediate__j = interm__j,
            intermediate_rvs__j = jax.random.normal,
            n_steps = n_mh_steps)

    source_rvs__j = jax.random.normal

    #[0, n_inter)
    betas = jnp.arange(n_inter).block_until_ready()

    do_ais__unorm2unorm__j : Callable[[PRNGKey], tuple[Samples, Weights]]
    do_ais__unorm2unorm__j = partial(do_ais__g, 
            n_samples = n_samples,
            n_inter = n_inter,
            betas = betas,
            target__j = target__j,
            source__j = source__j,
            intermediate__j = interm__j,
            source_rvs__j = source_rvs__j,
            transition_rule__j = trans_rule__j)

    return do_ais__unorm2unorm__j



def generic_ais(key : PRNGKey = None,
                n_samples : Dimension = None,
                m_inter: Dimension = None,
                init_params : Callable = None,
                f0_pdf : Callable = None,
                fj_pdf : Callable = None,
                T : Callable = None,
                init_params_kwargs : dict = None,
                f0_pdf_kwargs : dict = None,
                fj_pdf_kwargs : dict = None,
                T_kwargs : dict = None,
                ) -> tuple[DeviceArray]:
    """Warning, function must be partially compiled followed by a jax.jit compile
       else dictionary assignments will result in undefined behaviour.

       Designed for partial application of functions followed by jit compilation
       There are n_samples returned samples and weights.

       The default arguments are intialized to None because we want the function to fail
       loudly

       There are m_interpolating_dist

       let f_m be the dist of interest
       let fj_pdf be an interpolating dist
           fj_pdf(j, val)

       let T be the markov transition rule

       """
    def n_samples_loop_body(sample_index : int, n_samples_init_val : tuple[DeviceArray]):
        # Generate an initial point
        key, subkey = jax.random.split(n_samples_init_val['key'])
        n_samples_init_val['key'] = key
        xn = gen_xn(sample_index, subkey, inner_init, **gen_xn_kwargs) 
        logw = 0
        m_inter_init_val = {'xn':xn, 
                            'sample_index':sample_index, 
                            'n_samples_init_val':n_samples_init_val,
                            'logw':logw}

        m_inter_out_val = jax.lax.fori_loop(1, m_inter, m_inter_loop_body, m_inter_init_val)
        samples_n, log_weights_n = n_samples_init_val
        samples_n = samples_n.at[samples_index].set(m_inter_out_val['samples'])

        return samples, weights

    def m_inter_loop_body(m_inter_index : int, m_inter_init_val : dict):
        """Use the Markov transition rule T(j, **kwargs) to compute p(xj | xj-1)
           params:
           return:
             m_inter_val: dict. A dictionary containing the statefull information"""

        #Apply the markov transition kernal
        xn = T(m_inter_index, **m_inter_init_val, **T_kwargs)

        #Compute the pdf of x_{j} and x{j-1}
        yj =fj_pdf(m_inter_index, **m_inter_init_val, **fj_pdf_kwargs) 
        yj_1 = fj_pdf(m_inter_index -1, **m_inter_init_val, **fj_pdf_kwargs)
        #compute the log weight j
        logw_m = jnp.log(yj) - jnp.log(yj_1)
        logw = m_inter_init_val['logw']
        #multiply current weight with the previous one
        logw = logw + logw_m
        #set the new weight and xj 
        m_inter_init_val['logw'] = logw
        m_inter_init_val['xn'] = xn
        
        return m_inter_init_val

    params = init_params(**init_params_kwargs)

    # initiate the samples and the weights
    samples = jnp.zeros(n_samples)
    log_weights = jnp.ones(n_samples)

    # The outer loop over the number of samples
    samples, log_weights = jax.lax.fori_loop(0, n_samples, n_samples_loop_body, (samples, log_weights))
    return samples, log_weights

# AIS in the context of sqr models
def get_phi_tilde(phi : Matrix, gamma : float) -> Matrix:
    phi_diag = phi.diagonal()
    phi_tilde = phi * gamma[j]
    phi_tilde = nblib.set_diag(phi_tilde, phi_diag)
    return phi_tilde


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

def get_mean__j(samples : Array = None, weights : Array = None)-> float:
    """params:
        samples: 1d array
        weights: 1d array 
       return:
         mean: float"""
    return jnp.sum(samples * weights) / jnp.sum(samples) 
