"""
Jittable log probability mass functions
The log base is e
"""
import jax.scipy.stats as stats


def bernoulli(k, p: float, loc=0, /) -> float:
    """

    Positional only arguments
    Params:
      k : success / fail
      p : [0,1] the bernoulli probability
    Returns:
      log_probability_mass : float [0, 1]
    """
    log_probability_mass = stats.bernoulli.logpmf(k, p, loc)
    return log_probability_mass 


def betabinom(k: int, n: int, a: float, b:float , loc=0, /) -> float:
    """
    The log probability mass function of the beta binomial distribution

    Params:
      k : the number of successes
      n : the number of Bernoulli trials
      a : alpha Beta parameter
      b : beta  Beta parameter 
      loc : location

    Returns:
      log_probability_mass
    """
    log_probability_mass = stats.betabinom.logpmf(k, n, a, b, loc)
    return log_probability_mass 


def geom(k, p, loc=0, /):
    """
    The log probability mass function of the geometric distribution
    """
    return stats.geom.logpmf(k, p, loc)


def nbinom(k, n, p, loc=0, /):
    return stats.nbinom.logpmf(k, n, p, loc)


def poisson(k, mu, loc=0, /) -> float:
    """
    The log probability mass funciton of the Poisson distribution

    Params:

      k: int or array like
      mu: mean rate of events
      loc: location
    Returns:
      log_probability_mass : 
    """
    log_probability_mass = stats.poisson.logpmf(k, mu, loc)
    return log_probability_mass 
