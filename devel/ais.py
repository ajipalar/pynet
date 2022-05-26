# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pyext.src.ais as ais
#from pyext.src.ais import key, n_samples, n_inter, betas, x
import jax
import jax.random as jrandom
import jax.numpy as jnp
import jax.scipy as jsp


from jax import grad, vmap
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from functools import partial

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,10)
# -

n_samples = 10000
n_inter = 50
n_mh_steps = 10
mu = 99
sig = 7
do_ais__unorm2unorm__j = ais.do_ais__unorm2unorm__p(mu = mu, sig = sig, n_inter = n_inter,
                                                   n_samples = n_samples, n_mh_steps = n_mh_steps)

jdo_ais__unorm2unorm = jax.jit(do_ais__unorm2unorm__j)

key = jax.random.PRNGKey(10)
weights, samples = jdo_ais__unorm2unorm(key)



# +
mu, sigma, f_n, x0, n_inter, n_samples, betas, key = ais.ais_prelude()
y = []
x = jnp.arange(-10, 10, 0.1)
f_n_notebook = jax.scipy.stats.norm.pdf

mu_notebook = 90
sig_notebook = 5
x_notebook = jnp.arange(-200, 200, 1)
for i in x:
    ys = f_n(i, loc=0, scale=1)
    y.append(ys)


def grad_plot(x, f_n):
    plt.plot(x, vmap(f_n)(x), 'k--',
             x, vmap(grad(f_n))(x), '22',
             x, vmap(grad(grad(f_n)))(x), '-',
             x, vmap(grad(grad(grad(f_n))))(x), 'g-')

    plt.show()

grad_plot(x, f_n)
# -

plt.plot(x, vmap(f_n)(x), color='19')

# +
data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

plt.scatter('a', 'b', c='c', s='d', data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.show()
# -

grad_plot(x, jsp.stats.logistic.cdf)

jsp.stats.logistic.

samples, weights = ais.ais_example(mu_notebook, sig_notebook, n_samples=100, n_gibbs_steps=10)


def get_ais_mean(ais_samples, ais_weights):
    return 1/np.sum(ais_weights) * np.sum(ais_samples * ais_weights)


get_ais_mean(samples, weights)

# +

f_n = jax.scipy.stats.norm.pdf
p_f0 = partial(ais.f_0, mu=mu_notebook, sig=sig_notebook)
p_fj = partial(ais.f_j, f_0=p_f0, f_n=f_n)
# -

for i in np.arange(0, 1, 0.1):
    plt.plot(x_notebook, p_fj(x_notebook, i))

# +
#x = np.arange(mu_notebook - 4*sig_notebook, sig_notebook * 4 + mu_notebook, 1/4 * sig_notebook)
xmin = -10
xmax=mu_notebook + 4*sig_notebook
x = np.arange(xmin, xmax, 0.5)
plt.plot(x, p_f0(x))
plt.plot(x, f_n(x))
for i in np.arange(0, 1, 0.1):
    plt.plot(x, p_fj(x, i))
    
plt.show()
# -

plt.plot(x, p_fj(x, 0.5))

x = np.arange(-2, 2, 0.1)
plt.plot(x, jax.scipy.stats.norm.pdf(x))
plt.plot(x, jax.scipy.stats.norm.logpdf(x))

x = np.arange(xmin, 2*xmax, 1)
plt.plot(x, ais.log_f0(x, mu=mu_notebook, sig=sig_notebook), scaley=False)
plt.plot(x, 200*ais.f_0(x, mu=mu_notebook, sig=sig_notebook))

p_log_f0 = partial(ais.log_f0(mu=mu_notebook, sig=sig_notebook))
p_log_fn = 
for beta in np.arange(0, 1, 0.1):
    plt.plot(x, 200*ais.f_j(x, beta mu=mu_notebook, sig=sig_notebook))
    plt.plot(x, ais.log_fj(x, beta, log_))

plt.plot(x, ais.f_0(x, mu=mu_notebook, sig=sig_notebook))

# ?plt.plot

plt.hist(weights * samples)

a = 1/np.sum(weights) * np.sum(weights * samples)

a

1/len(weights) * np.sum(weights)

ais.f_j

jdo_ais = jax.jit(ais.do_ais)

okey = key
samples, weights = ais.do_ais(okey, n_samples, n_inter, betas, x)

#target N(5, 2)
#proposal N(0, 1)
import numpy as np
a = 1/jnp.sum(weights) *(jnp.sum(weights * samples))

samples, weights
