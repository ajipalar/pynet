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

from collections import namedtuple
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import pandas as pd
from pathlib import Path
from functools import partial
import itertools
from itertools import combinations
import re
import requests
import json
import scipy as sp
import scipy.stats
import sklearn
import sys
import time
import pyext.src.pynet_rng as rng
import timeit
from src.wishart_synthetic_benchmark import (
example)
import pyext.src.matrix as mat
import pyext.src.stats as stats

# Flags
MICRO_BENCHMARK_SAMPLING = False

# +
# Global Notebook Variables
key = jax.random.PRNGKey(13)
n = 4
shape = 1
dim = 4
df = 2
p = 4

# Image Generating Values From a 2-d Gaussian

xy = -0.9
xx = 1.
yy = 1.
cov = jnp.array([[xx, xy],
                 [xy, yy]])
N=1000

mean = jnp.zeros(2)
data = jax.random.multivariate_normal(key, mean, cov, shape=(N,))


# +
x_stats = stats.get_stats(data[:, 0])
y_stats = stats.get_stats(data[:, 1])

def scatter_plot(x, y, title=None, N=1000):
    assert len(x) == len(y)
    N = len(x)
    fig, axs = plt.subplots()
    if not title:
        title = f"Normal random variates rho {xy}\nN={N}"
    plt.plot(x, y, 'k.')
    plt.title(f"Normal random variates rho {xy}\nN={N}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim((-4, 4))
    plt.xlim((-3, 3))
    ax = fig.gca()
    ax.vlines(np.mean(x), ymin=-5, ymax=5)
    ax.hlines(np.mean(y), xmin=-3, xmax=3)


def margins_plot(x, y, title=None):
    assert len(x) == len(y)
    N = len(x)
    fig, axs = plt.subplots(2, 2)
    if not title:
        title = f"Normal random variates rho {xy}\nN={N}"
    ax = axs[0, 0]
    ax.plot(x, y, 'k.')
    ax.set_title(f"Normal random variates rho {xy}\nN={N}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim((-4, 4))
    ax.set_xlim((-3, 3))
    ax.vlines(np.mean(x), ymin=-5, ymax=5)
    ax.hlines(np.mean(y), xmin=-3, xmax=3)
    
    def helper(ax, arg, color='k', bins=20):
        ax.hist(arg, color=color, bins=bins)
    
    kwargs = {"color":"k", "bins":100 }
    ax = axs[0, 1]
    
    #transform = Affine2D().rotate_deg(90)
    #plot_extents = -5, 5, 0, 10
    #h = floating_axes.GridHelperCurveLinear(transform, plot_extents)
    #ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=h)
    helper(ax, x, **kwargs)
    fig.add_subplot(ax)
    
    ax = axs[1, 0]
    helper(ax, y, **kwargs)
    
def rprior(key, V, n=2, p=2):
    return rng.wishart(key, V=V, n=n, p=p)

def rprior_pred(key, V, n, p, n_replicates):
    keys = jax.random.split(key, 2)
    S = rprior(keys[0], V=V, n=n, p=p)
    mean = jnp.zeros(p)
    return jax.random.multivariate_normal(keys[1], mean, cov=S, shape=(n_replicates,))

def get_prior_pred(key, V, n_replicates=3, n_samples=100, n=2, p=2):
    Val = namedtuple("Val", "key data")
    data = jnp.zeros((n_samples, n_replicates, p))
    
    def body(i, val):
        # N x p
        key, k1 = jax.random.split(val.key)
        prior_pred = rprior_pred(k1, 
                                 V=V, 
                                 n=n,
                                 p=p,
                                 n_replicates=n_replicates)
        data = val.data.at[i].set(prior_pred)
        return Val(key, data)
    
    init = Val(key, data)
    val = jax.lax.fori_loop(0, n_samples, body, init)
    return val

def randPOSDEFMAT(key, p):
    A = jax.random.uniform(key, shape=(p, p))
    A = 0.5 * (A + A.T)
    A = A + jnp.eye(p)*p
    return A

def quad_plot(prior_mat_stat_df, K):
    Kstats = get_precision_matrix_stats(K)
    names = ["rowsum", "medians", "vars", "means", "mins", "maxs", "absdets"]

    Kstats = {names[i]:Kstats[i] for i in range(len(Kstats))}
    prior_mat_stat_df = prior_mat_stat_df[["means", "vars", "mins", "maxs"]]
    scale = 8
    bins=30
    w = 1 * scale
    h = 1 * scale
    cmap1 = "CMRmap"#"nipy_spectral" #"CMRmap"\
    cmap2 = "nipy_spectral"
    facecolor = "steelblue"#"pink"#"cornflowerblue"
    vlinecolor = "darkorange"#"aquamarine"#"orange"
    cbar_scale = 0.35

    fig, axs = plt.subplots(2, 2, layout="constrained")

    locator = {0:(0, 0), 1:(0, 1), 2:(1, 0), 3:(1, 1)}



    for i in range(4):
        ax = axs[locator[i]]
        col = prior_mat_stat_df.iloc[:, i]
        if i==0:
            col = n*col
        xlabel = f"n-{col}"f"{col.name}"
        ax.set_xlabel(f"{col.name}")
        label = f"prior" if i==3 else None
        ax.hist(col.values, bins=bins, label=label, facecolor=facecolor)
        ax.set_ylabel("Frequency")
        ylim = ax.get_ylim()

        label = f"Ground Truth" if i==3 else None
        ax.vlines(ymin=0, ymax=ylim[1], x=Kstats[col.name], color=vlinecolor, label=label)

    plt.suptitle(f"N={n_samples}")
    plt.rc("font", **font_rc)
    fig.set_figheight(h)
    fig.set_figwidth(w)
    fig.legend()
    plt.show()

scatter_plot(data[:, 0], data[:, 1])
# -

margins_plot(x, y, title="")

# +
scale = 1.
xy = -0.99
cov_prior = jnp.array([[scale, xy],
                       [xy, scale]])

assert mat.is_positive_definite(cov_prior)

n_samples = 1000
val = jax.jit(partial(get_prior_pred, V=cov_prior, n_samples=n_samples))(key)

x = np.ravel(val.data[:, :, 0])
y = np.ravel(val.data[:, :, 1])
scatter_plot(x, y, )


# +
# What is the largest matrix this will work on?

def try_sampling(p):
    print(f"Trying {p}")
    n = p
    cov_prior = jnp.eye(p)
    n_replicates = 3
    n_samples = 100
    kwargs = {'p': p, 'V':cov_prior, "n_replicates": n_replicates, "n_samples":n_samples,
              "n":n}

    val = jax.jit(partial(get_prior_pred, **kwargs))(key)
    print(f"Done! {val.data.shape}")

if MICRO_BENCHMARK_SAMPLING:
    for p in [24, 48, 96, 128, 256, 512, 1024]:
        try_sampling(p)

# +
# What about inverse Covariance Matrices?

"""
Synthetic Data Setup
  64 member pulldown
  4 replicates
  Correlation Structure
    view the image
  mean vector
  
  view the correlation matrix
  view the empircal correlation
  view the precision matrix

Model - Prior Predictive Check
    K ~ Wish
    Sigma = inv(K)
    D ~ N(0, Sigma)
    
Gaussian Model
Updated.
"""

# +
p = 64
k = jax.random.PRNGKey(234762834)
k, k1, k2 = jax.random.split(k, 3)
A = randPOSDEFMAT(k, p)
diag_idx = np.diag_indices(p)
A = A.at[diag_idx].set(jnp.sqrt(A[diag_idx]) + jnp.arange(p) / 2).block_until_ready()
A = np.array(A)

n_replicates = 4
mus = jax.random.normal(k1, shape=(p,))*5
data = jax.random.multivariate_normal(k2, mus, A, shape=(n_replicates,))

assert mat.is_positive_definite(A)

K = np.linalg.inv(A)

#K =  K / np.abs((np.max(K) - np.min(K))) # put two -1, 1
#K = K * 2

font_rc = {"size": 14, "family": "sans-serif"}

scale = 16
w = 1 * scale
h = 1 * scale
cmap1 = "CMRmap"#"nipy_spectral" #"CMRmap"\
cmap2 = "nipy_spectral"
cbar_scale = 0.35

fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios':[1, 1],
                                           'height_ratios':[1]}, layout="constrained")

plt.rc("font", **font_rc)
fig.set_figheight(h)
fig.set_figwidth(w)
ax = axs[0]


covim = ax.imshow(A, vmin=np.min(A), vmax=2*np.median(A), cmap=cmap1)
fig.colorbar(covim, ax=ax, location="left", shrink=0.35)
ax.set_title("Covariance Matrix")

ax = axs[1]
ax.set_title("Precision Matrix")
precim = ax.imshow(K, vmin=np.min(K), vmax=np.max(K), cmap=cmap2)
fig.colorbar(precim, ax=ax, location="right", shrink=cbar_scale)
plt.suptitle("Ground Truth", y=0.75)
plt.show()
# -

# ?plt.suptitle

# The covariance matrix $A$  
# The precision matrix $K=A^{-1}$
#
# In the covariance matrix the off diagonal elements are only dependant on columns X and Y
#
# #### Covariance Matrix $A$
# - Diagonals are the Variances
# - Off Diagonals are the Covariances (un-normalized correlation)
#
# #### Correlation Matrix corr:
# - The covariance matrix of standardized normal variates
# - The diagonals are 1
# - The off-diagonals are the correlations
#   
# #### Precision Matrix $K$
# $$p_{ij} = K_{ij}$$ Precision matrix off diagonal
# partial_correlation 
# $$\rho_{X_iX_j\cdot V}=-\frac{K_{ij}}{\sqrt{p_{ii}p_{ij}}}$$
#
#

# +

data = np.array(data)
data = data.T
data = pd.DataFrame(data=data)
data["means"] = data.apply(np.mean, axis=1)
data["stds"] = data.apply(np.std, axis=1)
data["vars"] = data.apply(np.var, axis=1)
data["medians"] = data.apply(np.median, axis=1)
diag_idx = np.diag_indices(p)
data["mu"] = np.array(mus)
data["Aii"] = np.array(A[diag_idx])
data["Kii"] = np.array(K[diag_idx])


# -

def ccscatter(x, y):
    rho = np.corrcoef(x.values, y.values)
    pearson = rho[0, 1]
    pearson = np.round(pearson, decimals=3)
    plt.scatter(x, y, label=f"rho {pearson}")
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.legend()
    plt.show()
    


ccscatter(data.mu, data.means)

ccscatter(data.mu, data.loc[:, 0])

ccscatter(data.Aii, data.vars)

ccscatter(data.Aii, data.stds)

ccscatter(data.Aii, data.Kii)


# #### Model 1
#  - 64 Independant Gaussians parameterized by their mean and variance
#  $$p(d_i | \mu_i, \tau_i) = N(\mu, \tau)$$
#  $$ \mu \sim N(\mu_0, 1)$$
#  $$ \tau \sim \text{Beta} $$
#  $$d_i \sim N(\mu, \tau) $$ 
#  
# #### Model 2
#
# $$ D \sim N(\mu, \Sigma^{-1})$$
#
# $$ \mu \sim N(\mu_0, \tau_0) $$
#
# $$ \Sigma^{-1} \sim \mathcal{W}_p(\nu, K_0)$$
#  - Data $D$ with $n$ replicates and $p$ nodes
#  - A $p$ variate Gaussian distribution with
#  - $\nu$ degrees of freedom
#  - $V$ scale matrix
#  - $K_0 = V^{-1}$
#  - $n$ observations
#  - $p$ variables
#  - $Z = D \cdot D^{T}$
#  $$ D \sim W_{p}(n + \nu, \space K_0 + Z^{-1}) $$
#  
# #### Prior Predicitve Check Metrics
#
# #### Prior Comparisons to Ground Truth
#
# $$ K \sim \mathcal{W}_p(\nu, K_0) $$
#
# - sum of the rows
# - sum of the columns
# - sum of the trace
# - min
# - max
#
# #####  p-variate metrics
# - mean values
# - medians
# - variances
# - min
# - max
#
# ##### Eigen-values

def simulate_from_prior(key, nu, K_0):
    return rng.wishart(key, K_0, nu, len(K_0))


# +
K_0 = jnp.ones(p) * 0.5 + jnp.eye(p)
n_samples = 1000
def sample_from_prior(key, nu, n_samples, K_0):
    Val = namedtuple("Val", "keys samples")
    samples = jnp.zeros((n_samples, p, p))
    keys = jax.random.split(key, num=n_samples)
    
    init = Val(keys, samples)
    
    def body(i, val):
        K = simulate_from_prior(val.keys[i], nu, K_0)
        samples = val.samples.at[i].set(K)
        val = Val(val.keys, samples)
        return val
    val = jax.lax.fori_loop(0, n_samples, body, init)
    return val

jss = jax.jit(partial(sample_from_prior, K_0=K_0, nu=p, n_samples=n_samples))

def get_precision_matrix_stats(K, n):
    Stats = namedtuple("Stats", "rowsum medians means vars mins maxs absdets sumEx")
    assert K.shape == (64, 64)
    

    return Stats([np.sum(x) for x in K], 
                 np.median(K), 
                 np.mean(K), 
                 np.var(K), 
                 np.min(K), 
                 np.max(K),
                 np.abs(sp.linalg.det(K)))


val = jss(key)

# +
mat_stats = [get_precision_matrix_stats(K) for K in val.samples]

def df_from_stats(stats, n):
    rowsum_s = np.array([i.rowsum for i in stats])
    med_s = np.array([i.medians for i in stats])
    var_s = np.array([i.vars for i in stats])
    mean_s = np.array([i.means for i in stats])
    min_s = np.array([i.mins for i in stats])
    max_s = np.array([i.maxs for i in stats])
    absdet = np.array([i.absdets for i in stats])
    
    
    rowsum_s = str(rowsum_s)
    
    data = {'rowsum': rowsum_s,
            'medians': med_s,
            "vars": var_s,
            "means": mean_s,
            "mins": min_s,
            "maxs": max_s,
            "absdets": absdet}
    
    return pd.DataFrame(data=data)

prior_mat_stat_df = df_from_stats(mat_stats)
#prior_mat_stat_df = prior_mat_stat_df.iloc[:, 1:-1]
# -

# Initial Model Results
quad_plot(prior_mat_stat_df, K)

# Try a new K0, reuse key
K0 = jnp.eye(p)
n_samples = 1000
jss = jax.jit(partial(sample_from_prior, K_0=K0, nu=p, n_samples=n_samples))
val = jss(key)
stats = [get_precision_matrix_stats(K) for K in val.samples]
df = df_from_stats(stats)
quad_plot(df, K0)

# Try against the standard wishart


sp.stats.wishart.pdf(K0, p, K0)

prior_mat_stat_df.hist(column="vars")

prior_mat_stat_df.hist("vars")

# +
# Are the samples Wishart Distributed?

# ?sp.stats.wishart.pdf

