import numpy as np

J=8 # number of schools
#estimated treatment of effects
y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
#standard error of estimates
sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

import numpyro
import numpyro.distributions as dist

#Eight schools example

def eight_schools(J, sigma, y=None):
    #Population treatment effect
    mu = numpyro.sample('mu', dist.Normal(0, 5))
    #Standard deviation in treatment effects
    tau = numpyro.sample('tau', dist.HalfCauchy(5))
    with numpyro.plate('J', J):
         theta = numpyro.sample('theta', dist.Normal(mu, tau))
         numpyro.sample('obs', dist.Normal(theta, sigma), obs=y)

from jax import random
from numpyro.infer import MCMC, NUTS

nuts_kernal = NUTS(eight_schools)
mcmc = MCMC(nuts_kernal, num_warmup=500, num_samples=1000)
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, J, sigma, y=y, extra_fields=('potential_energy',))
mcmc.print_summary()

pe = mcmc.get_extra_fields()['potential_energy']
print('Expected log joint density: {:.2f}'.format(np.mean(-pe)))
