"""
Jittable log probability mass functions
The log base is e
"""
import jax.scipy.stats as stats

bernoulli = stats.bernoulli.logpmf
betabinom = stats.betabinom.logpmf
geom = stats.geom.logpmf
nbinom = stats.nbinom.logpmf
poisson = stats.poisson.logpmf

