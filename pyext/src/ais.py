from pyext.src.typedefs import Array, PRNGKey
import pyext.src.PlotBioGridStatsLib as nblib
import jax 
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from typing import Callable

mu = 5
sigma = 2
f_n = jax.scipy.stats.norm.pdf


x = np.arange(5, 15, 0.1)
n_inter = 50
n_samples = 100
betas = np.linspace(0, 1, n_inter)
key = jax.random.PRNGKey(10)

def f_0(x, mu=mu, sigma=sigma):
    """Target distribution: \propto N(mu, sigma)"""
    return jnp.exp(-((x - mu) / sigma)**2)

def f_j(x, beta):
    """Intermediate distribution: interpolation between f_0 and f_n"""
    return f_0(x)**beta * f_n(x)**(1 - beta)

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

def do_ais(key, n_samples, n_inter, betas, x):
    """Perform annealed importance sampling as Neal using the N-steps metropolis algoirthm
    """
    samples = jnp.zeros(n_samples)
    weights = jnp.zeros(n_samples)
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

def ais_example():
    """The example from Augstinus Kristiadi's blog"""

    f_n = st.norm.pdf
    p_n = st.norm(0, 1)
    
    def f_0(x):
        return np.exp(-(x + 5)**2 / 2 / 2)
    
    def f_j(x, beta):
        return f_0(x)**beta + f_n(x)**(1 - beta)

    def T(x, f, n_steps=10):
        for t in range(n_steps):

            x_prime = x + np.random.randn()

            a = f(x_prime) / f(x)

            if np.random.rand() < a:
                x = x_prime

        return x

    x = np.arange(-10, 5, 0.1)
    n_inter = 60
    betas = np.linspace(0, 1, n_inter)
    n_samples = 600
    samples = np.zeros(n_samples)
    weights = np.zeros(n_samples)

    for t in range(n_samples):
        x = p_n.rvs()
        w = 1

        for n in range(1, len(betas)):
            x = T(x, lambda x: f_j(x, betas[n]), n_steps=5)

            w += np.log(f_j(x, betas[n])) - np.log(f_j(x, betas[n - 1]))

        samples[t] = x
        weights[t] = np.exp(w)

    return samples, weights
