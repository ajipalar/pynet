"""
Jittable log probability density functions
The log base is e
"""
import jax.scipy.stats as stats

beta = stats.beta.logpdf
cauchy = stats.cauchy.logpdf
chi2 = stats.chi2.logpdf
dirichlet = stats.dirichlet.logpdf
expon = stats.expon.logpdf
gamma = stats.gamma.logpdf
gennorm = stats.gennorm.logpdf
laplace = stats.laplace.logpdf
logistic = stats.logistic.logpdf
multivariate_normal = stats.multivariate_normal.logpdf
norm = stats.norm.logpdf
pareto = stats.pareto.logpdf
t = stats.t.logpdf
uniform = stats.uniform.logpdf

