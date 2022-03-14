import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np


pdf = jax.scipy.stats.norm.pdf
lpdf = jax.scipy.stats.norm.logpdf
cdf = jsp.stats.norm.cdf
lcdf = jsp.stats.norm.logpdf
rv = jax.random.normal

