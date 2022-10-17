"""
Jittable log probability density functions
The log base is e
"""
import jax.scipy.stats as stats


def beta(x, a, b, loc=0, scale=1, /):
    return stats.beta.logpdf(x, a, b, loc, scale)

def cauchy(x, loc, scale, /):
    return stats.cauchy.logpdf(x, loc, scale)

def chi2(x, df, loc=0, scale=1, /):
    return stats.chi2.logpdf(x, df, loc, scale)

def dirichlet(x, alpha, /):
    return stats.dirichlet.logpdf(x, alpha)

def expon(x, loc=0, scale=1, /):
    return stats.expon.logpdf(x, loc, scale)

def gamma(x, a, loc=0, scale=1, /):
    return stats.gamma.logpdf(x, a, loc, scale)

def gennorm(x, p, /):
    return stats.gennorm.logpdf(x, p)

def laplace(x, loc=0, scale=1, /):
    return stats.laplace.logpdf(x, loc, scale)

def logistic(x, /):
    return stats.logistic.logpdf(x)

def multivariate_normal(x, mean, cov, allow_singular=None, /):
    return stats.multivariate_normal.logpdf(x, mean, cov, allow_singular)

def norm(x, loc=0, scale=1, /):
    return stats.norm.logpdf(x, loc, scale)

def pareto(x, b, loc=0, scale=1, /):
    return stats.pareto.logpdf(x, b, loc, scale)


def t(x, df, loc=0, scale=1, /):
    return stats.t.logpdf(x, df, loc, scale)


def uniform(x, loc=0, scale=1, /):
    return stats.uniform.logpdf(x, loc, scale)
