import numpyro
import numpyro.distriutions as dist
import numpy as np

def linear(mu, sigma, y):
    sigma = numpyro.sample('sigma', dist.Normal(0, 1))
    mu = numpyro.sample('mu', dist.Normal(0, 1))
    obs = numpyro.sample('obs', dist.Normal(mu, sigma), obs=y)

y = np.random.randn(0, 1)


