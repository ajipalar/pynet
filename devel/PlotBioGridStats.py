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

# + language="bash"
# python --version

# + language="bash"
# conda env list

# +
from pyext.src.typedefs import (
    GeneID,
    Matrix,
    Vector
)

from pyext.src.PlotBioGridStatsLib import (
    jbuild,
    get_matrix_col_minus_s,
    prepare_biogrid,
    load_tip49_spec_counts_dataset,
    plot_col,
    plot_physical_experiments,
    find_idmapping_overlap,
    filter_biogrid,
    ProteinName,
    Any,
    DataFrame,
    Array
)

from pyext.src.jittools import is_jittable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from pathlib import Path
from functools import partial
from typing import Any, Callable, NewType
import graphviz
import inspect
import scipy

import pyext.src.PlotBioGridStatsLib as nblib
from sklearn.metrics import roc_curve, precision_recall_curve
# -

### Dev zone
import pyext.src.ais as ais
from functional_gibbslib import gibbsf, generic_gibbsf, generic_gibbs, gibbs
import functional_gibbslib as fg
key = jax.random.PRNGKey(7)
theta, phi, X, p, n= nblib.dev_get_dev_state_poisson_sqr()
s=1
x_i = X[:, 2]
eta1 = nblib.get_eta1(phi, s)
eta2 = nblib.get_eta2(theta, phi, x_i, s, p)

nblib.set_diag(phi, theta)

# +
eta1 = -2
eta2 = 11

t1 = jnp.sqrt(jnp.pi) * eta1 * jnp.exp( -eta2 ** 2 / (4 * eta1))
erf_arg = - eta2 / (2 * (jnp.sqrt(-1j) * jnp.sqrt(eta1)))
# -

key = jax.random.PRNGKey(7)
key, k1 = jax.random.split(key)
theta_exp = jnp.zeros(p)
phi_exp = nblib.get_exp_random_phi(k1, p)
for s in range(1, p+1):
    eta1_exp = nblib.get_eta1(phi_exp, s)
    eta2_exp = nblib.get_eta2(theta_exp, phi_exp, x_i, s, p)
    z_exp = nblib.Zexp(eta1_exp, eta2_exp)
    a_exp = nblib.Aexp(eta1_exp, eta2_exp)
    print(z_exp, eta1_exp, eta2_exp)
    assert jnp.log(z_exp) == a_exp



x = scipy.stats.expon.rvs(scale=np.arange(0, 10), size=(10, 100))

scale = np.zeros((100, 10))
scale[:, :] = np.arange(10)

fg.generic_gibbsf

scale


# +
def myfunc(x, *args, **kwargs):
    print(x)
    
def myhigherfunc(fargs=[], fkwargs={}):
    myfunc('Hi', *fargs, **fkwargs)
    
myhigherfunc(fargs=range(100), fkwargs={str(i):i**2 for i in range(10)})
# -

scipy.stats.expon.rvs(scale=np.arange(10), size=10, random_state=2)



x

jax.scipy.stats.expon.pdf(x)

a = [1, 2, 3]
a.reverse()
a

# +
# matmul test

a = np.arange(20)
b = a * np.arange(20)
# -



# load biogrid data
dpath = Path("../data/biogrid/BIOGRID-ALL-4.4.206.tab3.txt")
biogrid = prepare_biogrid(dpath)

plot_col(biogrid,'Experimental System Type' )

plot_col(biogrid, 'Experimental System', topn=12)

plot_physical_experiments(biogrid)

# load in the spectral counts data
spec_counts_df = load_tip49_spec_counts_dataset()

# returns None
find_idmapping_overlap(biogrid, spec_counts_df)

#Filters out non tip49 entires
biogrid = filter_biogrid(spec_counts_df, biogrid)
tip49biogrid = biogrid
del biogrid

plot_physical_experiments(tip49biogrid)

plot_col(tip49biogrid, 'Experimental System')

spec_counts_df

tip49biogrid

query = tip49biogrid['Official Symbol Interactor A'].iloc[0:10]        

# Annealed Importance Sampling of Poisson SQR Model
# Index  $i$ ranges from $1$ to $N$ where $N$ is the number of weights $w^{(i)}$
# and index $j$ ranges from $1$ to $n$ where $n$ is the number of intermediate distributions
#

# +
spec_counts = []
for i in spec_counts_df['Spec']:
    i = i.split('|')
    i = list(i)
    for j in i:
        j = int(j)
        spec_counts.append(j)
        
spec_counts = np.array(spec_counts)
plt.hist(spec_counts, bins=1000)
#plt.xticks([0, 50, 100, 150])
plt.xlim(0, 20)
plt.show()
# -


