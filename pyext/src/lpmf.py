"""
Jittable log probability mass functions
The log base is e
"""
import jax.scipy.stats as stats


def bernoulli(k, p, loc=0, /):
    return stats.bernoulli.logpmf(k, p, loc)


def betabinom(k, n, a, b, loc=0, /):
    return stats.betabinom.logpmf(k, n, a, b, loc)


def geom(k, p, loc=0, /):
    return stats.geom.logpmf(k, p, loc)


def nbinom(k, n, p, loc=0, /):
    return stats.nbinom.logpmf(k, n, p, loc)


def poisson(k, mu, loc=0, /):
    return stats.poisson.logpmf(k, mu, loc)
