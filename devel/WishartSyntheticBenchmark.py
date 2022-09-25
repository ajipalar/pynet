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
import pyext.src.matrix as mat
import pyext.src.stats as stats

from src.wishart_synthetic_benchmark import (
    ccscatter,
    df_from_stats,
    get_precision_matrix_stats,
    get_prior_pred,
    helper_vline_hist,
    margins_plot,
    quad_plot,
    randPOSDEFMAT,
    rprior,
    rprior_pred,
    sample_from_prior,
    scatter_plot,
    simulate_from_prior,
    try_sampling
    
)
# -

# Flags
MICRO_BENCHMARK_SAMPLING = False
eig10k = False
PAIR_PLOTS = False

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
# -


x_stats = stats.get_stats(data[:, 0])
y_stats = stats.get_stats(data[:, 1])

scatter_plot(data[:, 0], data[:, 1], xy=xy)

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
# -

# What is the largest Matrix this will work on?
if MICRO_BENCHMARK_SAMPLING:
    for p in [24, 48, 96, 128, 256, 512, 1024]:
        try_sampling(key, p)

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
assert np.alltrue(A[diag_idx] > 0)
assert mat.is_positive_definite(A)

n_replicates = 4
mus = jax.random.normal(k1, shape=(p,))*5
data = jax.random.multivariate_normal(k2, mus, A, shape=(n_replicates,))

assert mat.is_positive_definite(A)

K = np.linalg.inv(A)
assert np.alltrue(K[diag_idx] > 0)
assert np.alltrue(K[diag_idx] > 0)
assert mat.is_positive_definite(K)
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

# +
K_0 = jnp.ones(p) * 0.5 + jnp.eye(p)
n_samples = 1000

jss = jax.jit(partial(sample_from_prior, K_0=K_0, nu=p, p=p, n_samples=n_samples))

val = jss(key)

mat_stats = [get_precision_matrix_stats(K, n) for K in val.samples]

prior_mat_stat_df = df_from_stats(mat_stats, n)
#prior_mat_stat_df = prior_mat_stat_df.iloc[:, 1:-1]
# -

# Initial Model Results
suptitle = f"N={n_samples}\nFrom {p}-variate Wishart"
quad_plot(prior_mat_stat_df, K, n, n_samples, font_rc=font_rc, p=p, suptitle=suptitle)



# Try a new K0, reuse key
p=64
n=64
ground_truth = K_0
K0 = 1/n * K_0
n_samples = 1000
jss = jax.jit(partial(sample_from_prior, K_0=K0, nu=p, p=p, n_samples=n_samples))
val = jss(key)
stats = [get_precision_matrix_stats(K, n) for K in val.samples]
df = df_from_stats(stats, n)
quad_plot(df, ground_truth, n_samples=n_samples,
         n=n,
         font_rc=font_rc, p=p)

# +
# Try a new K0, reuse key

p=64
n=64
K0 = 1/n * K
n_samples = 10000
jss = jax.jit(partial(sample_from_prior, K_0=K0, nu=p, p=p, n_samples=n_samples))
val = jss(key)
stats = [get_precision_matrix_stats(K, n) for K in val.samples]
df = df_from_stats(stats, n=n)
# -

quad_plot(df, K0, n=n, n_samples=n_samples, font_rc=font_rc, p=p)

# +
# Lets look at the top 4 eigenvalues

if eig10k:
    ground_truth = sp.linalg.eigh(K, eigvals_only=True, check_finite=True)

    eigs = np.zeros((len(val.samples), p))
    for i in range(len(val.samples)):
        eigs[i] = sp.linalg.eigh(val.samples[i], eigvals_only=True, check_finite=True)
    
# -

if PAIR_PLOTS:
    m = 8
    scale = 16
    w = 1 * scale
    h = 1 * scale
    bins=30
    hcolor="steelblue"
    vcolor="darkorange"
    fig, axs = plt.subplots(m, m, layout="constrained")
    fig.set_figheight(h)
    fig.set_figwidth(w)

    count = -1
    for i in range(m):
        for j in range(m):
            count +=1
            ax = axs[i, j]
            eigvals = eigs[:, count]
            vx = ground_truth[count]
            ymin = 0

            vlabel = f"K eigenvalues" if count == m*m-1 else None
            hlabel = f"prior samples" if count == m*m-1 else None

            #helper_vline_hist(ax, vx, ymin, ymax, eigvals, vlabel, hlabel,
                             #vcolor, hcolor, bins, ylabel=None, xlabel=None)



            ax.hist(eigvals, bins=bins, label=hlabel, facecolor=hcolor)
            ymax = ax.get_ylim()[1]
            ax.vlines(x=truth, ymin=0, ymax=ymax, color=vcolor)

            #ax.scatter(x, y)
        plt.suptitle(f"N={n_samples}")
    plt.show()

# +
# A simpler Case

cov = np.array([[1., -0.2],
                [-0.2, 1.]])

cov_inv = sp.linalg.inv(cov)
nu = 2
n=2
p = 2
K0 = (1/nu)*cov_inv
n_samples = 1000

key = jax.random.PRNGKey(22)
samples = sample_from_prior(key, nu, p, n_samples, K0)

# +
t1 = A
t2 = K
A = cov
K = cov_inv
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
A = t1
K = t2

# +
mat_stats = [get_precision_matrix_stats(K, n, p=p) for K in samples.samples]

prior_mat_stat_df = df_from_stats(mat_stats, n)
# -

quad_plot(df, cov_inv, n=n, n_samples=n_samples, font_rc=font_rc, p=p)

# +
ground_truth = sp.linalg.eigh(cov_inv, eigvals_only=True, check_finite=True)

eigs = np.zeros((len(samples.samples), p))
for i in range(len(samples.samples)):
    eigs[i] = sp.linalg.eigh(samples.samples[i], eigvals_only=True, check_finite=True)

# +
m = 2
scale = 6
w = 1.5 * scale
h = 1 * scale
bins=30
hcolor="steelblue"
vcolor="darkorange"

font_rc = {"size": 16, "family": "sans-serif"}
fig, axs = plt.subplots(1, 2, layout="constrained")
fig.set_figheight(h)
fig.set_figwidth(w)

count = -1
axs[0].set_ylabel("Frequency")
for i in range(m):
    count +=1
    ax = axs[i]
    eigvals = eigs[:, count]
    vx = ground_truth[count]
    ymin = 0

    vlabel = f"K eigenvalues" if count == m*m-1 else None
    hlabel = f"prior samples" if count == m*m-1 else None

    #helper_vline_hist(ax, vx, ymin, ymax, eigvals, vlabel, hlabel,
                     #vcolor, hcolor, bins, ylabel=None, xlabel=None)


    truth = ground_truth[count]
    ax.hist(eigvals, bins=bins, label=hlabel, facecolor=hcolor)
    ymax = ax.get_ylim()[1]
    ax.vlines(x=truth, ymin=0, ymax=ymax, color=vcolor)

    s = u"\u03BB"
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    s = s + str(count + 1)
    s = s.translate(SUB)
    ax.set_xlabel(s)
    #ax.scatter(x, y)
    
    plt.suptitle(f"N={n_samples}")
plt.show()

# +
# More Complex 3 x 3

cov = np.array([[1., 0., 0.2],
                [0., 0.3, -0.5],
                [0.2, -0.5, 1.3]])

assert mat.is_positive_definite(cov)

cov_inv = sp.linalg.inv(cov)
nu = 3
n=3
p = 3
K0 = (1/nu)*cov_inv
n_samples = 1000

key = jax.random.PRNGKey(22)

samples = sample_from_prior(key, nu, p, n_samples, K0)

# +
mat_stats = [get_precision_matrix_stats(K, n, p=p) for K in samples.samples]

prior_mat_stat_df = df_from_stats(mat_stats, n)
# -

quad_plot(prior_mat_stat_df, cov_inv, n=n, n_samples=n_samples, font_rc=font_rc, p=p)

# +
ground_truth = sp.linalg.eigh(cov_inv, eigvals_only=True, check_finite=True)

eigs = np.zeros((len(samples.samples), p))
for i in range(len(samples.samples)):
    eigs[i] = sp.linalg.eigh(samples.samples[i], eigvals_only=True, check_finite=True)
# -

assert ground_truth.shape == (3,)
assert eigs.shape == (n_samples, 3)

# +
m = 3
scale = 6
w = 1.5 * scale
h = 1 * scale
bins=30
hcolor="steelblue"
vcolor="darkorange"

font_rc = {"size": 16, "family": "sans-serif"}
fig, axs = plt.subplots(1, m, layout="constrained")
fig.set_figheight(h)
fig.set_figwidth(w)

count = -1
axs[0].set_ylabel("Frequency")
for i in range(m):
    count +=1
    ax = axs[i]
    eigvals = eigs[:, count]
    vx = ground_truth[count]
    ymin = 0

    vlabel = f"K eigenvalues" if count == m*m-1 else None
    hlabel = f"prior samples" if count == m*m-1 else None

    #helper_vline_hist(ax, vx, ymin, ymax, eigvals, vlabel, hlabel,
                     #vcolor, hcolor, bins, ylabel=None, xlabel=None)


    truth = ground_truth[count]
    ax.hist(eigvals, bins=bins, label=hlabel, facecolor=hcolor)
    ymax = ax.get_ylim()[1]
    ax.vlines(x=truth, ymin=0, ymax=ymax, color=vcolor)

    s = u"\u03BB"
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    s = s + str(count + 1)
    s = s.translate(SUB)
    ax.set_xlabel(s)
    #ax.scatter(x, y)
    
    plt.suptitle(f"N={n_samples}")
plt.show()


# +
# More Complex 4x4
def set_up_synth_exp(key, nu, cov,):
    SynthExper = namedtuple("SynthExp", "cov cov_inv nu n p K0 n_samples key samples")
    assert mat.is_positive_definite(cov)

    cov_inv = sp.linalg.inv(cov)
    n=len(cov)
    p = len(cov)
    K0 = (1/nu)*cov_inv
    n_samples = 1000

    key = jax.random.PRNGKey(22)

    samples = sample_from_prior(key, nu, p, n_samples, K0)
    return SynthExper(cov, cov_inv, nu, n, p, K0, n_samples, key, samples)

cov = np.array([[1.1,  0.0, 0.1, 1.],
                [0.,  1.2, 1., 0.],
                [0.1, 1.0,  1.1,0.],
                [1.,  0.,  0., 1.]])


def check_cov(m):
    assert np.alltrue(m[np.diag_indices(len(m))] > 0), f"fail : diag"
    assert np.alltrue(cov == cov.T)
    assert mat.is_positive_definite(m), f"fail pos"
    
def do_quad_plot(exp, font_rc={"size": 16}):
    mat_stats = [get_precision_matrix_stats(S, exp.n, p=exp.p) for S in exp.samples.samples]
    prior_mat_stat_df = df_from_stats(mat_stats, exp.n)
    quad_plot(prior_mat_stat_df, exp.cov_inv, n=exp.n, n_samples=exp.n_samples, font_rc=font_rc, p=exp.p)

key = jax.random.PRNGKey(4)
ex4 = set_up_synth_exp(key, nu, cov)
do_quad_plot(ex4)


# -

def do_gridplot(
    exp,
    scale = 6,
    w = 1.5 * scale,
    h = 1 * scale,
    bins=30,
    hcolor="steelblue",
    vcolor="darkorange",

    font_rc = {"size": 16, "family": "sans-serif"}, 
    check_finite=True,
    decomposition="eigh"):
    
    

    if decomposition == "eigh":
        def decomp(x):
            return sp.linalg.eigh(x, eigvals_only=True, check_finite=check_finite)
    
    elif decomposition == "svd":
        def decomp(x):
            U, s, VH = sp.linalg.svd(x)
            return s
    elif decomposition == "prec":
        def decomp(x):
            return x[np.diag_indices(len(x))]
        
    ground_truth = decomp(exp.cov_inv)
    
    eigs = np.zeros((len(exp.samples.samples), exp.p))
    
    
    for i in range(len(exp.samples.samples)):
        eigs[i] = decomp(exp.samples.samples[i])
    
    
    
    assert len(exp.cov) % 2 == 0, f"rank cov is odd"
    m = int(np.sqrt(len(exp.cov)))
    fig, axs = plt.subplots(m, m, layout="constrained")
    fig.set_figheight(h)
    fig.set_figwidth(w)

    count = -1
    axs[0, 0].set_ylabel("Frequency")
    

    
    for i in range(m):
        for j in range(m):
            count +=1
            ax = axs[i, j]
            eigvals = eigs[:, count]
            vx = ground_truth[count]
            ymin = 0

            vlabel = f"K eigenvalues" if count == m*m-1 else None
            hlabel = f"prior samples" if count == m*m-1 else None

            #helper_vline_hist(ax, vx, ymin, ymax, eigvals, vlabel, hlabel,
                             #vcolor, hcolor, bins, ylabel=None, xlabel=None)


            truth = ground_truth[count]
            ax.hist(eigvals, bins=bins, label=hlabel, facecolor=hcolor)
            ymax = ax.get_ylim()[1]
            ax.vlines(x=truth, ymin=0, ymax=ymax, color=vcolor)
            
            if decomposition == "eigh":
                s = u"\u03BB"

            elif decomposition == "svd":
                s = u"\u03C3"
            
            elif decomposition == "prec":
                s = f"p"

            SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
            s = s + str(count + 1)
            s = s.translate(SUB)
            ax.set_xlabel(s)
            #ax.scatter(x, y)

            plt.suptitle(f"N={n_samples}")
    #plt.legend()
    plt.show()

do_gridplot(ex4)

# +
key = jax.random.PRNGKey(1123123)
key, k1 = jax.random.split(key, 2)
p = 16
nu = p
A = jax.random.uniform(key, shape=(p, p)).block_until_ready()

A = A @ A.T
A = A / (np.sqrt(A) @ np.sqrt(A))
A = A + np.eye(p)
A = np.array(A)
A[0, 3] = 0
A[3, 0] = 0

check_cov(A)
exp = set_up_synth_exp(k1, nu, A) # check invertible, pos def

check_cov(exp.cov_inv) # check inverse

assert np.alltrue(np.isnan(exp.K0)==False)
assert np.alltrue(np.isinf(exp.K0)==False)
assert np.alltrue(np.isnan(exp.samples.samples)==False)
# -

do_gridplot(exp)

do_quad_plot(exp)

# +
"""
Let's Do an Example with an 8 x 8 matrix
"""
p=16 # The length of the date vector
nu = p
n_trial = 4 # The number of AP-MS trials
factor = 4

# Generate the Ground Truth Network
key = jax.random.PRNGKey(22)
keys = jax.random.split(key, 10)
A = jax.random.bernoulli(keys[0], shape=(p, p))
diag_idx = np.diag_indices(p)
A = np.tril(A) + np.tril(A).T
A = np.array(A)
A[diag_idx] = 0
A = np.array(A, dtype=int)

# Generate the K_theta, the simulate known precicion matrix

K_theta = jax.random.uniform(keys[1], minval=-1., maxval=1.,  shape=(p, p))
K_theta = np.array(K_theta)
K_theta /= factor
K_theta[np.where(A == 0)] = 0
K_theta = np.tril(K_theta, k=-1) + np.tril(K_theta, k=-1).T
K_theta[diag_idx] = 1 + jax.random.normal(keys[2], shape=(p,))/4

check_cov(K_theta)
#assert np.sum(K_theta[diag_idx]) == p

# +
def ground_truth_pair_plot(A, K, title1="", title2="", cmap1="nipy_spectral", cmap2="CMRmap", factor=1.):
    font_rc = {"size": 14, "family": "sans-serif"}

    scale = 16
    w = 1 * scale
    h = 1 * scale
    cbar_scale = 0.35

    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios':[1, 1],
                                               'height_ratios':[1]}, layout="constrained")

    plt.rc("font", **font_rc)
    fig.set_figheight(h)
    fig.set_figwidth(w)
    ax = axs[0]


    covim = ax.imshow(A, vmin=np.min(A), vmax=2*np.median(A), cmap=cmap1)
    fig.colorbar(covim, ax=ax, location="left", shrink=cbar_scale)
    ax.set_title(title1)
    #ax.legend()

    ax = axs[1]
    ax.set_title(title2)
    
   # if type(cmap2) == str:
   #     cmap2 = matplotlib.colors.Colormap(cmap2)
    K_plot = K.copy()
    
    K_plot[np.diag_indices(len(K_plot))] = np.max(np.tril(K_plot, k=-1))
    
    precim = ax.imshow(K_plot, vmin=np.min(K_plot), cmap=cmap2)
    
    #bounds = [-1/factor, 1/factor]
    #cnorm = matplotlib.colors.BoundaryNorm(bounds, cmap2.N)

    fig.colorbar(precim, ax=ax, location="right", shrink=cbar_scale)
    #plt.suptitle(, y=0.75)
    plt.show()
    
cmap1 = matplotlib.colors.ListedColormap(['w', 'steelblue'])


ground_truth_pair_plot(A, K_theta, title1="A", 
                       title2="K" + u"\u03B8"
                       , cmap1=cmap1,
                       cmap2 = "RdBu")

check_cov(K_theta)


# -

def get_exp(key, nu, p, n, n_samples, V, K_theta, K0, n_trial):
    keys = jax.random.split(key)
    Ks = sample_from_prior(keys[0], nu, p, n_samples, V)
    K_theta = np.array(K_theta)
    K0 = np.array(K0)
    SynthExper = namedtuple("SynthExp", "cov cov_inv nu n p K0 n_samples key samples")
    exp = SynthExper(np.array(jsp.linalg.inv(K_theta)), K_theta, nu, n_trial, p, K0, n_samples, keys[1], Ks)
    return exp

# +
# p length of the data vector
#assert np.sum(K_theta[diag_idx]) == p, f"{p, np.sum(K_theta[diag_idx])}"


K0 = np.ones(shape=(p, p)) * -0.1
#diag_K0 = jax.random.uniform(keys[2], minval=0, maxval=0.01, shape=(p, ))
K0[np.diag_indices(p)] = 2
check_cov(K0)
n_samples = 1000
V = K0 / p

check_cov(V)


# -

def get_exp(key, nu, p, n, n_samples, V, K_theta, K0, n_trial):
    keys = jax.random.split(key)
    Ks = sample_from_prior(keys[0], nu, p, n_samples, V)
    K_theta = np.array(K_theta)
    K0 = np.array(K0)
    SynthExper = namedtuple("SynthExp", "cov cov_inv nu n p K0 n_samples key samples")
    exp = SynthExper(np.array(jsp.linalg.inv(K_theta)), K_theta, nu, n_trial, p, K0, n_samples, keys[1], Ks)
    return exp


nu = p-1
exp = get_exp(key, nu, p, n, n_samples, V, K_theta, K0, n_trial)


do_gridplot(exp, check_finite=False, decomposition="prec")

do_gridplot(exp, decomposition="svd")

# +
K0 = K_theta
diag = np.diag_indices(p)
#diag_K0 = jax.random.uniform(keys[2], minval=0, maxval=0.01, shape=(p, ))

check_cov(K0)
n_samples = 1000



nu = p - 2

V = K0 / (nu)

check_cov(V)

exp = get_exp(key, nu, p, n, n_samples, V, K_theta, K0, n_trial)
# -

do_gridplot(exp, decomposition="prec")


do_gridplot(exp, decomposition="svd")

# +
# Conclusion - setting a prior of K_theta is worse. Perhaps because of the zeros

# +
K0 = np.eye(p)
diag = np.diag_indices(p)
#diag_K0 = jax.random.uniform(keys[2], minval=0, maxval=0.01, shape=(p, ))

check_cov(K0)
n_samples = 10000



check_cov(V)
nu = p - 2
V = K0 / (nu)

exp = get_exp(key, nu, p, n, n_samples, V, K_theta, K0, n_trial)
# -

do_gridplot(exp, decomposition="prec")

do_gridplot(exp, decomposition="svd")

do_gridplot(exp)

jsp.linalg.eigh(K_theta, eigvals_only=True)

K_theta[diag]

# +
# Lets look at prior accuracy and precision
# TP TN FP FN
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve


def get_accuracies_and_precisions(A, exp):
    nedges = sp.special.binom(p, 2)

    AUROC = np.zeros(n_samples, dtype=float) # Average AUROC
    AUPRC = np.zeros(n_samples, dtype=float)
    for i, Ksamp in enumerate(np.array(exp.samples.samples)):
        assert Ksamp.shape == (p, p)
        minval = np.min(np.tril(Ksamp, k=-1))
        maxval = np.max(np.tril(Ksamp, k=-1))
        #assert minval < 0 < maxval
        d = maxval - minval

        LT = A[np.tril_indices(p, k=-1)]
        LTK = Ksamp[np.tril_indices(p, k=-1)]
        LT = np.array(LT)
        LTK = np.array(LTK) 

        #assert np.all(LT == LTK)
        y_test = np.ravel(LT)
        y_score = np.ravel(LTK)
        #assert np.all(y_test == y_score)
        #assert np.sum(y_test) > 0
        
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_score)
        precision, recall, thresholds_prec = sklearn.metrics.precision_recall_curve(y_test, y_score)

        AP = sklearn.metrics.average_precision_score(y_test, y_true)
        area_under_roc = sklearn.metrics.auc(fpr, tpr)
        area_under_prc = sklearn.metric.auc(recall, precision)

        AUROC[i] = area_under_roc
        AUPRC[i] = area_under_prc
        
        #assert APs.shape == (n_samples, )
        #assert AAUROCs.shape == (n_samples, )
    
    return AUROC, AUPRC, thresholds, thresholds_prec


# +
# Generate the Ground Truth Network
key = jax.random.PRNGKey(22)
keys = jax.random.split(key, 10)
A = jax.random.bernoulli(keys[0], shape=(p, p))
diag_idx = np.diag_indices(p)
A = np.tril(A) + np.tril(A).T
A = np.array(A)
A[diag_idx] = 0
A = np.array(A, dtype=int)

# Generate the K_theta, the simulate known precicion matrix
del K_theta
K_theta = jax.random.uniform(keys[1], minval=-1., maxval=1.,  shape=(p, p))
K_theta = np.array(K_theta)
K_theta /= factor

K_theta[0:10] += 0.05
K_theta[4:7] += 0.3
K_theta[4, 3] = 1
K_theta[9, 2] = -1
K_theta[7:15, 7:15] -= 0.2
K_theta[5:9, 5:9] += 0.1

K_theta[np.where(A == 0)] = 0
K_theta = np.tril(K_theta, k=-1) + np.tril(K_theta, k=-1).T
K_theta[diag_idx] = 1 + jax.random.normal(keys[2], shape=(p,))/4


K_theta[0, 0] = 10
K_theta[1, 1] = 1
K_theta[2, 2] = 2
K_theta[3, 3] = 5
K_theta[4, 4] = 12
K_theta[5, 5] = 231
K_theta[6, 6] = 10121
K_theta[7, 7] = 100232
K_theta[8, 8] = 999999
K_theta[9, 9] = 1283828
K_theta[10, 10] = 4
K_theta[11, 11] = 4
K_theta[12, 12] = 5
K_theta[13, 13] = 9
K_theta[14, 14] = 20
K_theta[15, 15] = 40

#K_theta = sp.linalg.inv(K_theta)


#K_theta[10, 10] = 7
K0 = K_theta
diag = np.diag_indices(p)
#diag_K0 = jax.random.uniform(keys[2], minval=0, maxval=0.01, shape=(p, ))

check_cov(K0)
n_samples = 1000

V = K0 / p

check_cov(V)

nu = p - 1
exp = get_exp(key, nu, p, n, n_samples, V, K_theta, K0, n_trial)
# -

cmap = "twilight_shifted"
cmap = "PuOr"
cmap = "seismic"
#cmap = "ocean" # "slategrey" # "steelblue" #whitesmoke
cmap1 = matplotlib.colors.ListedColormap(['w', 'lightskyblue'])
ground_truth_pair_plot(A, K_theta, title1="A", 
                       title2="K" + u"\u03B8"
                       ,cmap1=cmap1,
                       cmap2 = cmap)

do_gridplot(exp, decomposition="prec")

do_gridplot(exp, decomposition="svd")

assert A.shape == (p, p)
assert np.sum(A[diag]) == 0
accs, aps, t, tp = get_accuracies_and_precisions(A, exp)

plot_accuracies_and_precisions(accs, aps)

# +
K0 = np.eye(p)
K0[np.diag_indices(p)] = K_theta[np.diag_indices(p)]
check_cov(K0)
n_samples = 1000

V = K0 / p

check_cov(V)

nu = p - 1
exp = get_exp(key, nu, p, n, n_samples, V, K_theta, K0, n_trial)
# -

do_gridplot(exp, decomposition="prec")

do_gridplot(exp, decomposition="svd")

assert A.shape == (p, p)
assert np.sum(A[diag]) == 0
assert np.alltrue(A == A.T)
#check_cov(A)
assert np.sum(A) != 0
aacs, aps, t, tp = get_accuracies_and_precisions(A, exp)
plot_accuracies_and_precisions(AAUROCs, APs)

# +
# Lets look at prior accuracy and precision
# TP TN FP FN
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve


def get_accuracies_and_precisions(A, exp):
    nedges = sp.special.binom(p, 2)

    AAUROCs = np.zeros(n_samples, dtype=float) # Average AUROC
    APs = np.zeros(n_samples, dtype=float)
    for i, Ksamp in enumerate(np.array(exp.samples.samples)):
        assert Ksamp.shape == (p, p)
        minval = np.min(np.tril(Ksamp, k=-1))
        maxval = np.max(np.tril(Ksamp, k=-1))
        #assert minval < 0 < maxval
        d = maxval - minval

        LT = A[np.tril_indices(p, k=-1)]
        LTK = Ksamp[np.tril_indices(p, k=-1)]
        LT = np.array(LT)
        LTK = np.array(LTK) 

        #assert np.all(LT == LTK)
        y_test = np.ravel(LT)
        y_score = np.ravel(LTK)
        #assert np.all(y_test == y_score)
        #assert np.sum(y_test) > 0
        
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_score)
        precision, recall, thresholds_prec = sklearn.metrics.precision_recall_curve(y_test, y_score)

        AP = sklearn.metrics.average_precision_score(precision, recall)
        area_under_roc = sklearn.metrics.auc(fpr, tpr)

        APs[i] = AP
        AAUROCs[i] = area_under_roc
        
        #assert APs.shape == (n_samples, )
        #assert AAUROCs.shape == (n_samples, )
    
    return AAUROCs, APs, thresholds, thresholds_prec


# +
i = 0
Ksamp = As.samples[0]
assert Ksamp.shape == (p, p), f"{Ksamp.shape}, {p}"
minval = np.min(np.tril(Ksamp, k=-1))
maxval = np.max(np.tril(Ksamp, k=-1))
#assert minval < 0 < maxval
d = maxval - minval

LT = A[np.tril_indices(p, k=-1)]
LTK = Ksamp[np.tril_indices(p, k=-1)]
LT = np.array(LT)
LTK = np.array(LTK) 


y_test = np.ravel(LT)
y_score = np.ravel(LTK)

fpr, tpr, thresholds = roc_curve(y_test, y_score)
precision, recall, thresholds_prec = precision_recall_curve(y_test, y_score)

AP = sklearn.metrics.average_precision_score(y_test, y_score)
area_roc = sklearn.metrics.auc(fpr, tpr)

APs[i] = AP
AAUROCs[i] = area_roc
# -

area_roc

# +
# Dummy test for plotting accuracies and precisions
SynthExper = namedtuple("SynthExp", "cov cov_inv nu n p K0 n_samples key samples")
exp_test = exp
As = np.zeros((n_samples, p, p))
Samples = namedtuple("Samples", "samples")

for i in range(n_samples):
    As[i] = A
As = Samples(As)
exp_test = SynthExper(A, A, nu, n, p, A, n_samples, key, As)
accs_test, precs_test, ts, tps = get_accuracies_and_precisions(A, exp_test)
plot_accuracies_and_precisions(accs_test, precs_test)
# -

accs.shape

A



# ?plot_accuracies_and_precisions

def plot_accuracies_and_precisions(aacs, precs):
    bins=20

    plt.subplot(121)

    plt.hist(aacs, bins=bins)
    plt.xlabel("AUROC")
    plt.ylabel("Frequency")
    plt.subplot(122)
    plt.hist(precs, bins=bins)
    plt.xlabel("Average Precision")
    plt.suptitle(f"Prior Accuracy and precision")
    plt.show()


AAUROCs, APs = get_accuracies_and_precisions(exp)

plot_accuracies_and_precisions(AAUROCs, APs)

plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % area,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()

# Lets do a misfit prior
K0 = jax.random.exponential(key, shape=(p, p))
K0 = K0 * np.eye(p)
K0 = K0 * np.arange(-8, 8, 1)
K0 = np.array(K0)
K0[np.diag_indices(p)] += 10
V = K0 / (nu)
#check_cov(V)

plt.imshow(V, cmap=cmap)

exp = get_exp(key, nu, p, n, n_samples, V, K_theta, K0, n_trial)

do_gridplot(exp, decomposition="prec")

do_gridplot(exp, decomposition="svd")

accs, precs = get_accuracies_and_precisions(exp)

plot_accuracies_and_precisions(accs, precs)

# +
example_string = "A0B1C2D3E4F5G6H7I8J9"

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

print(example_string.translate(SUP))
print(example_string.translate(SUB))
# -
matplotlib.colors.Colormap(cmap1)


A


class FunctionSpace:
    """
    A vector space of functions from x in X to y in Y
    F: X -> Y
    """
    def __init__(self, f):
        if type(f) == FunctionSpace:
            self.f = f.f
        else:
            self.f = f
        
    def __call__(self, x):
        return self.f(x)
    
    def __add__(self, g):
        return FunctionSpace(lambda x: self.f(x) + g(x)) # be a member of the vector space
    
    def __radd__(self, g):
        return FunctionSpace(lambda x: self.f(x) + g(x))
    
    def __sub__(self, g):
        return FunctionSpace(lambda x: self.f(x) - g(x))
    
    def __rsub__(self, g):
        return FunctionSpace(lambda x: g(x) - self.f(x))
    
    def __mul__(self, lam):
        # scalar multiplication
        return FunctionSpace(lambda x: self.f(x) * lam)   
    
    def __rmul__(self, lam):
        return FunctionSpace(lambda x: self.f(x) * lam)


# +
fs = FunctionSpace

def f(x):
    return np.sqrt(x + 1)

def g(x):
    return x**2 + 2

def h(x):
    return x**2 + 9 / 3

def z(x):
    return x + 1

def q(x):
    return x + 1


# +
# Check our Assumptions

f = fs(f)
h = fs(h)
g = fs(g)

xs = [0, 2., np.array([[0, 1, 2],
                       [1., 1., -3.]])]

for x in xs:
    # Commutative over addition
    np.testing.assert_almost_equal( (f + g)(x), (g + f)(x)), f"{x}"

    # Scalar Multiplication
    np.testing.assert_almost_equal(0 * (f + h)(x), 0 * (h + f)(x))
    np.testing.assert_almost_equal(123 * (f + h)(x), (123 * f + 123 * h)(x))

    # Ascociative

    assert (h + (f + g))(3) == ((h + f) + g)(3)

    assert (f - f)(123498723489) == 0 # Additive Inverse

