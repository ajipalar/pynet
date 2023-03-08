"""
This module contains the source code for the corresponding notebook
X.X-model-training.ipynb where X.X is the version
"""
from collections import namedtuple
from functools import partial
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import scipy as sp
import re
import os
import shutil
import sys
import requests
import subprocess
import json
import Bio

# custom modules
import cullin_benchmark_test as cb_test
from dev_model_proto import *
import get_cullin_e3_ppi_from_pdb70 as ppi_pdb70
import lpdf
import ulpdf
import model_proto as mp
import pynet_rng
import HHBlits

# Note book globals
cb_path = Path("../../data/raw/cullin_e3_ligase/")
cb = cb_test.CullinBenchMark(cb_path)
cb_train = cb_test.CullinBenchMark(cb_path)

#The bait id for the training data
bait_id = "CBFBwt_MG132"
sel1 = cb.data["Bait"] == bait_id
#The saint score threshold
sel2 = cb.data["SaintScore"] >= 0.3

cb.data = cb.data.loc[sel1, :]
title = u"Bait - CBF\u03B2"
title2 = title + u"\nSAINT score \u2265 0.3"
nb_style = "ggplot"

cb_train.data = cb_train.data.loc[(sel1 & sel2), :]


n_examples = 1000
p = 0.1
key = jax.random.PRNGKey(13)

cb.parse_spec_counts()
cb_train.parse_spec_counts()

def get_n_edges_and_disconnectivity(key, p, n_examples): 
    n_edges = np.zeros(n_examples)
    disconnectivity = np.zeros(n_examples)

    for i in range(n_examples):
        key, k1 = jax.random.split(key)
        A = mp.proposal(k1, 44, n_examples, p=p)
        A = np.array(A, dtype=int)
        nedges = mp.nedges(A)
        n_edges[i] = nedges
        disconnectivity[i] = mp.d(A, np.arange(44))

    return n_edges, disconnectivity



# 1 Define M

# 1 Define M

N = 44
n_prey = 43

M = {"As": np.zeros((N, N), dtype=int),
          "Sigma_inv_s": np.eye(n_prey),
          "lambda_s": 0.5,
          "alpha_s": 0.5,
          "Cs": np.arange(44),
          "ts": 0.0,
          "mus": np.zeros(n_prey)}
def get_saint_threshold(cb, thresh):
    sel = cb.data["SaintScore"] >= thresh
    return cb.data.loc[sel, :]
    

def get_Ss(cb_train):
    Ss = np.array(cb_train.data["SaintScore"])
    assert Ss.shape == (43,), Ss.shape
    return Ss

def get_D(cb_train):
    D = {"ds": cb_train.data.loc[:, ["r1", "r2", "r3", "r4"]].values}
    assert D["ds"].shape == (43, 4)
    return D 

def get_I(cb_train):
    Ss = get_Ss(cb_train)
    I = {"Ss": Ss}
    return I

def get_Y(cb_train):
    D = get_D(cb_train)
    Y = D["ds"]
    assert Y.shape == (43, 4)
    return Y

def get_centered_Y(cb_train):
    Y = get_Y(cb_train)
    mean = np.mean(Y, axis=1).reshape((43, 1))
    centered_Y = Y - mean
    var = np.var(Y, axis=1).reshape((43, 1))
    assert centered_Y.shape == (43, 4)
    return centered_Y

def get_Scatter(cb_train, diag_scaler=0.1):
    """
    Params:
      diag_scaler: I * diag_scaler is added to the scatter matrix to
        make it positive definite 
    """
    centered_Y = get_centered_Y(cb_train)
    Scatter = centered_Y @ centered_Y.T
    assert Scatter.shape == (43, 43)
    Scatter = Scatter + jnp.eye(43) * diag_scaler
    return Scatter

def get_Sigma_inv_prior_guess(cb_train, diag_scaler=0.1):
    """

    """
    Scatter = get_Scatter(cb_train, diag_scaler) 
    assert np.all(1 - np.isnan(sp.linalg.cholesky(Scatter)))
    Sigma_inv_prior_guess = jsp.linalg.inv(Scatter)
    return Sigma_inv_prior_guess

def get_scale_matrix_V(cb_train, scatter_diag_scaler):
    """
    The scatter matrix is S0 + I*scatter_diag_scaler
    """
    Sigma_inv_prior_guess = get_Sigma_inv_prior_guess(cb_train, scatter_diag_scaler)
    assert len(Sigma_inv_prior_guess) == 43
    V = (1/43) * Sigma_inv_prior_guess
    return V
    


# 2 Define the score function

# p(M| D, I) prop p(D | M, I)p(M|I)
#   p(M|I)
#     p(As|Cs, lambda_s)p(Cs|Ss, ts)p(ts)p(mus)


V = get_scale_matrix_V(cb_train, scatter_diag_scaler=0.1)
z = mp._move_Sigma_inv(key, V, 43, 43)

# Prior predictive distribtuion for this scatter matrix


def log_pdf_ts(ts):
    return sp.stats.norm.logpdf(ts, loc=0.6, scale=0.08)

g_s = mp.log_pdf__M_D_I_restraint_builder(Ss, log_pdf_ts)

def log_prior(M):
    """
    Returns an array of log scores
      - As | Cs, lambda_s
      - Cs | Ss, ts
      - ts | I
    """
    return g_s(M["As"], M["Cs"], M["ts"], M["mus"], M["lambda_s"])

    #  p(D| M, I)

def log_like(D, M) -> float:
    s = 0
    Y = D['ds']
    f = mp.log_pdf_yrs__mus_Sigma_inv_s
    mu_s = M['mus']
    Sigma_inv = M['Sigma_inv_s']
    As = M['As']
    alpha_s = M['alpha_s']
                                                                    
    for i in range(4):
        s += f(Y[:, i], mu_s, Sigma_inv)
        log_Sigma_inv = mp.log_pdf_Sigma_inv_s__As_alpha_s(Sigma_inv,
          As,
          alpha_s)
    return np.array([s, log_Sigma_inv])

def log_score(D, M):

    lp = log_prior(M)
    ll = log_like(D, M)
    
    return np.sum(lp) + np.sum(ll), ll, lp

def initialize_model(cb_train, 
                     alpha_s=0.5, 
                     lambda_s=0.5,
                     As=None,
                     Sigma_inv_s=None,
                     Cs=None,
                     mus=None,
                     ):
    """
    Initialize M0 for training
    """

    assert len(set(cb_train.data["Prey"])) == 43
    N = 44
    n_prey = 43

    edges = ["As"]
    nuisance = ["Sigma_inv_s", "Cs", "ts", "mus"]
    constants = ["lambda_s", "alpha_s"]

    if not As:
        As = np.zeros((N, N), dtype=int)

    if not Sigma_inv_s:
        Sigma_inv_s = np.eye(n_prey)

    if not Cs:
        Cs = np.arange(N)

    if not mus:
        mus = np.zeros(n_prey)

    M0 = {"As": As, 
          "Sigma_inv_s": Sigma_inv_s,
          "Cs": Cs,
          "lambda_s": lambda_s,
          "alpha_s": alpha_s,
          "ts": ts,
          "mus": mus}

    return M0


def plot_saint_fdr(cb, title: str):
    plt.style.use(nb_style)
    plt.title(title)
    plt.plot(cb.data['SaintScore'], label='SAINT score')
    plt.plot(cb.data['BFDR'],  label="FDR")
    plt.legend()
    plt.xlabel('Prey ID')

def plot_edge_density(rseed, p, n_examples, nbins=25):
    key = jax.random.PRNGKey(rseed)
    n_edges, _ = get_n_edges_and_disconnectivity(key, p, n_examples)
    plt.style.use(nb_style) # reference the module level global variable
    plt.hist(n_edges, bins=nbins)
    plt.xlabel('n edges')
    plt.ylabel('frequency')
    plt.title(f'Bernoulli edge probability p={p}\nN={n_examples}')
    plt.show()

def plot_disconnectivity(rseed, p, n_examples, nbins=20, bin_range=(0, 10)):
    key = jax.random.PRNGKey(rseed)
    _, disconnectivity = get_n_edges_and_disconnectivity(key, p, n_examples)
    plt.style.use(nb_style)
    plt.hist(disconnectivity, bins=20, range=bin_range)
    plt.xlabel('n disconnected prey')
    plt.ylabel('frequency')
    plt.title('Bernoulli proposal p=0.5')
    plt.show()

def plot_hyper_param_predictive_check(sim_values, 
                                      real_values,
                                      real_operator,
                                      n_prey,
                                      nsamples,
                                      xlabel,
                                      title,
                                      w=4, 
                                      h=3, 
                                      sim_label='Simulated data',
                                      real_label='Real data',
                                      style="classic"
                                      ):

    plt.figure(figsize=(w, h))
    plt.style.use(style)
    plt.hist(sim_values, bins=30, label=sim_label)
    for i in range(4):

        if i == 3:
            plt.vlines(real_operator(real_values[:, i]), 0, 100, 'b', label=real_label)
        else:
            plt.vlines(real_operator(real_values[:, i]), 0, 100, 'b')
        plt.xlabel(xlabel)
    plt.legend()
    plt.title(title)
    plt.ylabel('Frequency')
    plt.show()

# Prior predictive distribtuion for this scatter matrix

class MoverTraining:
    def __init__(self, 
                 cb_train, 
                 rseed=13, 
                 nsamples=1000,
                 diag_scaler=0.1,
                 style=nb_style, 
                 fig_height = 6,
                 fig_width = 6):

        self.cb_train = cb_train
        self.rseed = rseed
        self.nsamples = nsamples
        self.diag_scaler = diag_scaler
        self.style=style
        self.fig_height = fig_height
        self.fig_width = fig_width

        self.key = jax.random.PRNGKey(rseed)
        self.keys = jax.random.split(self.key, nsamples)

        self.mean_sim = np.zeros(nsamples)
        self.var_sim = np.zeros(nsamples)
        self.sum_sim = np.zeros(nsamples)

        self.Scatter = get_Scatter(cb_train, diag_scaler=diag_scaler)

        for i in range(nsamples):
            y_sim = jax.random.multivariate_normal(self.keys[i], mean=np.zeros(43), cov=self.Scatter)
            self.mean_sim[i] = np.mean(y_sim)
            self.var_sim[i] = np.var(y_sim)
            self.sum_sim[i] = np.sum(y_sim)

        self.centered_Y = get_centered_Y(cb_train)


        self.mean_plot = partial(plot_hyper_param_predictive_check, 
                            real_values=self.centered_Y,
                            real_operator=np.mean,
                            n_prey=43,
                            nsamples=nsamples,
                            xlabel=f'Centered spectral count mean', 
                            title=u'Constant \u03A3 matrix\nN samples ' + f'{self.nsamples}',
                            style=self.style)
        
        self.var_plot = partial(plot_hyper_param_predictive_check, 
                           real_values=self.centered_Y,
                           real_operator=np.var,
                           n_prey=43,
                           nsamples=nsamples,
                           xlabel=f'Centered spectral count variance',
                           title=u'Constant \u03A3 matrix\nN samples ' + f'{self.nsamples}',
                           style=self.style)
        
        self.sum_plot = partial(plot_hyper_param_predictive_check, 
                           real_values=self.centered_Y,
                           real_operator=np.sum,
                           n_prey=43,
                           nsamples=nsamples,
                           xlabel=f'Cenetered spectral count sum',
                           title=u'Constant \u03A3 matrix\nN samples ' + f'{self.nsamples}',
                           style=self.style)

    def plot_mean_sim(self):
        self.mean_plot(self.mean_sim)

    def plot_var_sim(self):
        self.var_plot(self.var_sim)

    def plot_sum_sim(self):
        self.sum_plot(self.sum_sim)
    
    
Ss = get_Ss(cb_train) 

def q_cond_lpdf(Mi, Mj, p, wish_dof) -> float:
    """
    This funciton gives the log probability of moving from one point to another point
    according to the proposal distribution for the CBFB training set.

    The log conditional density of the proposal distribution q
    gives the probability of moving from one state to another state

    q(Mi| Mi)

    Params:
      Ma  : The model dictionary at the ith step 
      Mb  : The model dictionary at the jth step
      p   : The dimensionality of the precision matrix. For the CBFB training
            p=43. p is provided so that this function may be jit compiled

      wish_dof : aka nu. The wishart degrees of freedom where nu > p - 1
                 and p is the length of the scatter matrix
    """

    p1 = lpdf.norm(mu_i, loc=mu_j, scale=1)
    p2 = lpdf.uniform(ts_i) # don't really have to evaluate this. 
    V_j = Sigma_inv_s_j / p 
    p3 = ulpdf.wishart(cov_inv_i, wish_dof, V_j, p) 

    # Moving edges required selecting n_edges to move 
    # and flipping them independantly according to a Bernoulli trial
    # let x_i be a list of n_edges and x_j be another list of n_edges
    # p(x) is Binomial(n_edges, p) however we are interested in p(x_i|x_j)
    # Let y = x_i == x_j
    # 0 <= sum(y) <= n_edges
    # TBD 




def move_model(key, 
        M0, 
        flip_probability,
        move_n_edges,
        wish_dof,
        all_possible_edges_As,
        len_As):
    """
    This function must be jit compiled. Otherwise python will reference the
    dictionary incorectly. 
                    
                        
    Take a step in parameter space
    Params:
      key : jax PRNG key
      M0  : The model dictionary at the current time step
      flip_probability : the Bernoulli probability of flipping an edge
      move_n_edges: The number of Bernoulli edge flip trials to perform 
      wish_dof : the wishart degrees of freed aka nu
      all_possible_edges_As : an (m, 2) array of all possible edges from As
          where m = len_As * (len_As - 1) // 2
      len_As   : the length of adjacency matrix As 
    Returns:
      M1  - The proposed model dictionary
    """
    n_prey = len_As - 1

    keys = jax.random.split(key, 3)
    mu_s_0 = M0['mus']
    Sigma_inv_s = M0['Sigma_inv_s']
    As_0 = M0['As']
                                                                                       
    # Move mu_s_1 | mu_s_0

    
    mu_s_1 = mu_s_0 + jax.random.normal(keys[0], shape=(n_prey,))
      
    # Move ts_1  | ts_0

    ts_1 = jax.random.uniform(keys[1])
    # Move Simga_inv_s_1 | Sigma_inv_s
    Sigma_inv_s_1 = pynet_rng.wishart(keys[2], V=Sigma_inv_s / n_prey, n=wish_dof, p=43)

    # Move As_1  | As_0  # The number of flips is distributed binomially(n_flips, flip_probability, move_n_edges)
    As_1 = mp.flip_adjacency__j(key, As_0, flip_probability, all_possible_edges_As, move_n_edges, len_As)

    # alpha_s is constant
    # lambda_s is constant
    M0['mus'] = mu_s_1
    M0['As'] = As_1
    M0['Sigma_inv_s'] = Sigma_inv_s_1
    M0['ts'] = ts_1

    return M0



def stepf(i, keys, n, mean_sim, var_sim, sum_sim):
    Sigma_inv_i = pynet_rng.wishart(keys[i + 1], V=V, n=n, p=43)
    Sigma_i = jsp.linalg.inv(Sigma_inv_i)
    y_sim = jax.random.multivariate_normal(keys[i], mean=jnp.zeros(43), cov=Sigma_i)
                    
    x_ = jnp.mean(y_sim)
    v_ = jnp.var(y_sim)
    s_ = jnp.sum(y_sim)
                                    
    mean_sim = mean_sim.at[i].set(x_)
    var_sim = var_sim.at[i].set(v_)
    sum_sim = sum_sim.at[i].set(s_)
    return i, keys, n, mean_sim, var_sim, sum_sim

def do_sampled_sigma_experiment(nsamples, n, rseed):
    """
    Used to understand the effect of the degrees of freedom nu on the movers
    Params:
      nsamples:
      n:  nu the degrees of freedom
      rseed
    """

    key = jax.random.PRNGKey(rseed)
    keys = jax.random.split(key, nsamples + 1)
    
    step = jax.jit(stepf)

    m = jnp.zeros(nsamples)
    v = jnp.zeros(nsamples)
    ss = jnp.zeros(nsamples)
    values = 0, keys, n, m, v, ss
    for i in range(nsamples):
        _, *x = values
        values = i, *x
        _, *x = step(*values)
        values = i, *x

    _, keys, _, m, v, ss = values
    return m, v, ss

def get_log_score_fun(log_score, D):
    return partial(log_score, D=D)

def mh_step(key, M, log_prob_fun, proposal):
        
    key, key2 = jax.random.split(key)
            
    M1 = jit_move(key, M)
    log_score_0, *_ = log_prob_fun(M=M)
                        
    log_score_1, *_ = log_prob_fun(M=M1)
                                
                                    
    M, accepted, log_score = jit_step(key2, M, log_score_1, log_score_0, move_model)
                                            
    return M, u <= a, log_score


def _mh_step__j(key2, M, log_score_1, log_score_0):
        
    a = jnp.exp(log_score_1 - log_score_0)
            
    u = jax.random.uniform(key2)
                    
    M, log_score = jax.lax.cond(u > a, lambda : (M, log_score_0), lambda :(M1, log_score_1))
                            
    return M, u <= a, log_score


def sample_MH(key, x, ulpdf, q, q_cond_lpdf):
    """
    Perfrom a single local step in parameter space according to the
    proposal distribution q. Accept or reject based on the Metropolis Criterion.
    q is assumed to be asymetric q(x1|x0) != q(x0|x1). Therefore the conditional density for q
    is required to account for detailed balance.
    Params:
      key - the jax PRNGKey
      x   - the model variable
      ulpdf the unormalized log probability density funciton
      q     the proposal distribtuion
      q_cond_lpdf 
    """
    keys = jax.random.split(key)
    x1 = q(keys[0], x)

    alpha = min(1, np.exp(ulpdf(x1) + q_cond_lpdf(x1, x) - ulpdf(x) - q_condf_lpdf(x, x1))) 
    u = jax.random.uniform(keys[1])
    accepted = u <= alpha
    if accepted:
        x = x1
    return accepted, x


def build_re_mcmc_chain(rseed, n_mcmc_steps, betas, ulpdf, ulpdf_q, x0, q, swap_interval):
    n_replicas = len(betas)

    if k > 0 and k % swap_interval == 0:
        # Attempt RE Swap
        ...

    else:
        # Do local sampling
        ...

def plot_wishart_dof_effect(x1,
    x2,
    x_const,
    mover_training,
    nbins=30, 
    xmin=-100, 
    xmax=100, 
    density=True,
    alpha=0.5, 
    style='classic', 
    label1='n=43',
    label2='n=50', 
    ylabel='density', 
    xlabel='Centered mean spectral count',
    title='Effect of n (degrees of freedom) on the sample mean',
    real_label = 'real data',
    ymin=0, 
    ymax = 0.08,
    textstr=u'Bait: CBF\u03B2\nn-prey=43\nn-replicates=4\nN simulated replicates=1000',
    textx=-90,
    texty=0.06,
    vlines_operator=np.mean):
      
    hist = partial(plt.hist, bins=nbins, range=(xmin, xmax), density=density, alpha=alpha)

    plt.style.use(style)

    hist(x1, label=label1)
    hist(x2, label=label2)
    hist(x_const, label='const')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)

    plt.vlines(vlines_operator(mover_training.centered_Y, axis=0), label=real_label, ymin=ymin, ymax=ymax)
    plt.text(textx, texty, textstr)
    plt.legend()
    plt.show()

class CullinBenchMarkAnalysis:
    def __init__(self, df):

        self.data = df
        self.prey_set = set(df["Prey"])
        self.bait_set = set(df["Bait"])
        self.nprey = len(self.prey_set)
        self.nbait = len(self.bait_set)
        self.uniprot_re = (
            r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}"
        )
        self.is_uniprot = re.compile(self.uniprot_re)
        self.vif_uid = "P69723"

        nuid = 0
        not_uid = []
        for prey in self.prey_set:
            if self.is_uniprot.match(prey):
                nuid += 1
            else:
                not_uid.append(prey)

        self.nuid = nuid
        self.not_uid_set = set(not_uid)
        assert len(self.not_uid_set) == len(not_uid)
        self.nnuid = len(self.not_uid_set)

        elob_in_prey = False
        cbfb_in_prey = False
        cul5_in_prey = False

        for prey_gene in set(self.data["PreyGene"]):
            if prey_gene == "ELOB_HUMAN":
                elob_in_prey = True
            if prey_gene == "PEBB_HUMAN":
                cbfb_in_prey = True
            if prey_gene == "CUL5_HUMAN":
                cul5_in_prey = True

        self.elob_in_prey_set = elob_in_prey
        self.cbfb_in_prey_set = cbfb_in_prey
        self.cul5_in_prey_set = cul5_in_prey

        self.all_bait_in_prey_set = self.elob_in_prey_set and self.cbfb_in_prey_set and self.cul5_in_prey_set

        if self.all_bait_in_prey_set:
            self.n_total_proteins = self.nprey
        else:
            assert False, "baits are not in prey set"

        self.n_possible_edges = self.n_total_proteins * (self.n_total_proteins - 1) // 2

        viral_prey_set = []
        potential_viral_prey = ["envpolyprotein",
                                "gagpolyprotein",
                                "nefprotein",
                                "polpolyprotein",
                                "revprotein",
                                "tatprotein",
                                "vifprotein"]
        
        potential_mouse_prey = ["IGHG1_MOUSE"]
        mouse_prey = []


        for prey in self.not_uid_set:
            if prey in potential_viral_prey:
                viral_prey_set.append(prey)
            elif prey in potential_mouse_prey:
                mouse_prey.append(prey)
            else:
                assert False, f"Unmapped prey {prey}"

        self.viral_prey_set = set(viral_prey_set)
        self.nviral = len(viral_prey_set)

        self.mouse_prey = mouse_prey
        self.nmouse_prey = len(mouse_prey)

        self.vif_in_prey_set = False 
        self.tat_in_prey_set = False 
        self.rev_in_prey_set = False 
        self.poly_in_prey_set = False 
        self.nef_in_prey_set = False 
        self.gag_in_prey_set = False 
        self.env_in_prey_set = False 
    
        self.IGH1_in_prey_set = False
        
        n_viral_bool = 0

        if "vifprotein" in self.viral_prey_set:
            self.vif_in_prey_set = True
            n_viral_bool +=1
        if "tatprotein" in self.viral_prey_set:
            self.tat_in_prey_set = True
            n_viral_bool +=1
        if "revprotein" in self.viral_prey_set:
            self.tat_in_prey_set = True
            n_viral_bool +=1
        if "nefprotein" in self.viral_prey_set:
            self.tat_in_prey_set = True
            n_viral_bool +=1
        if "polpolyprotein" in self.viral_prey_set:
            self.tat_in_prey_set = True
            n_viral_bool +=1
        if "gagpolyprotein" in self.viral_prey_set:
            self.tat_in_prey_set = True
            n_viral_bool +=1
        if "envpolyprotein" in self.viral_prey_set:
            self.tat_in_prey_set = True
            n_viral_bool +=1

        assert n_viral_bool == len(self.viral_prey_set), (n_viral_bool, self.viral_prey_set)


        organisms = []

        

        for prey_gene in self.data["PreyGene"]:
            split_gene = prey_gene.split("_")
            if prey_gene in self.viral_prey_set:
                organisms.append("VIRAL")
            elif split_gene[1] == "HUMAN":
                organisms.append("HUMAN")

            elif split_gene[1] == "MOUSE":
                organisms.append("MOUSE")

            elif split_gene[1] not in  ("HUMAN", "MOUSE"):
                assert False, f"unexpected prey {prey_gene}"

            else:
                assert False, f"missed case {prey_gene}"

        self.data.loc[:, "organism"] = organisms

        self.nviral_conditions = sum(self.data["organism"] == "VIRAL")
        self.nmouse_conditions = sum(self.data["organism"] == "MOUSE")
        self.nhuman_conditions = sum(self.data["organism"] == "HUMAN")


        self.nconditions = len(self.data)

        assert self.nviral_conditions + self.nmouse_conditions + self.nhuman_conditions == self.nconditions



        self.nviral_prey = len(self.viral_prey_set)

        human_prey_set = set(self.data.loc[self.data["organism"] == "HUMAN", "Prey"])
        self.human_prey_set = human_prey_set
        self.nhuman_prey = len(self.human_prey_set)


        mouse_prey_set = set(self.data.loc[self.data['organism'] == "MOUSE", "Prey"])
        self.mouse_prey_set = mouse_prey_set
        self.nmouse_prey = len(mouse_prey_set)


        assert self.nviral_prey + self.nhuman_prey + self.nmouse_prey == self.nprey







    def create_uid_column(self):
        """
        Creates a column that is guaranteed to be uniprot ids
        Maps viral proteins to the uniprot ids
        """
        uids = []
        for prey_name in self.data.loc[:, "Prey"]:
            if self.is_uniprot.match(prey_name):
                uids.append(prey_name)
            elif prey_name == "vifprotein":
                uids.append(self.vif_uid)
            else:
                assert False, f"{prey_name} not a uniprot ID"

        self.data.loc[:, "UID"] = uids

    def __repr__(self):
        def h(x):
            return '{:,}'.format(x)

        def r(x):
            return np.round(x, 2)

        def p(a, b):
            return (a / b ) * 100

        def z(a, b):
            return f"{h(a)}    ({r(p(a, b))})%"

        nprey = h(self.nprey)
        nbait = h(self.nbait)
        nuid =  h(self.nuid)
        nnuid = h(self.nnuid)
        ntotalproteins = h(self.n_total_proteins)
        npp = h(self.n_possible_edges)

        s=f"""n conditions    {self.nconditions}
  viral    {z(self.nviral_conditions, self.nconditions)}    
  human    {z(self.nhuman_conditions, self.nconditions)} 
  mouse    {z(self.nmouse_conditions, self.nconditions)}

n unique prey: {nprey}
  viral    {z(self.nviral_prey, self.nprey)} 
  human    {z(self.nhuman_prey, self.nprey)}
  mouse    {z(self.nmouse_prey, self.nprey)}


n bait    {nbait}
n uid     {nuid}
n nuid    {nnuid}

----Human Bait-----
ELOB in prey set     {self.elob_in_prey_set}
CBFB in prey set     {self.cbfb_in_prey_set}
CUL5 in prey set     {self.cul5_in_prey_set}

---Viral Proteins----
VIF  in prey set     {self.vif_in_prey_set}
TAT  in prey set     {self.tat_in_prey_set}
REV  in prey set     {self.rev_in_prey_set}
POLY in prey set     {self.poly_in_prey_set}
NEF  in prey set     {self.nef_in_prey_set}
GAG_POLY in prey set {self.gag_in_prey_set}
ENV_POLY in prey set {self.env_in_prey_set}
------Mouse---------
IGHG1 in prey set    {self.IGH1_in_prey_set}

All bait in prey set {self.all_bait_in_prey_set}

n total proteins       {ntotalproteins}
n possible pairs       {npp}

        """
        return s


def assemble_ground_truth(
    dir_name: str,
    uid_list: list[str],
    min_seq_id_with_query: float,
    min_seq_coverage_with_query: float,
    min_e_value: float,
    hhblits_cpu=2,
    direct_interaction_criterion='500A_buried_surface_area',
    pdb70_path="~/databases/pdb70/pdb70",
    overwrite_dir=False,
    make_dir=False,
    get_fastas_1=False,
    get_hhr_2=False,
    get_pdb_pair_ids_3=False,
    hhr_prob=50.0
    ):
    """
    Assembles a ground truth set based on various criteria
    
    Params:
      dir_name: the name of the benchmark analysis. Also the
                the name of the directory to write to
      uid_list: a list of uniprot ids
      pdb70_path: the path to pdb70
      
      HHBlits Params:
        see https://github.com/soedinglab/hh-suite/wiki
        min_seq_id_with_query: the minimal sequence identity to the query
                               From the HHBlits manual - hhblits can detect
                               homology below the twilight zone of 20%
                               go to 10%
        min_seq_coverage_with_query: the minimal sequence coverage to query
        min_e_value:                 the minimal e value
      
      direct_interaction_criterion: The criterion for defining direct interactions

      overwrite_dir: overwrite the directory if it already exists
      make_dir     : make the directory
      get_fastas_1 : get the fasta sequences from uniprot 
      get_hhr_2    : run hhblits on the fastas
      hhr_prob     : HHBlits outputs a list of PDB entires with an ascociated
                     Hits should be considered when 
                        1) hit has > 50% probability
                        2) hit has > 30% probability and is in top 3

                        Probability accounts for secondary structure and e-value does not

                     
    """
    nproteins = len(uid_list)
    assert nproteins > 1
    assert 0 <= min_seq_id_with_query <= 100
    assert 0 <= min_e_value <= 1
    npossible_interactions = nproteins * (nproteins - 1) // 2

    if overwrite_dir:
        os.rmdir(dir_name)

    if make_dir:

        os.mkdir(dir_name)
        os.chdir(dir_name)
        os.mkdir("uniprot_seqs")
        os.mkdir("hhblits_out")

    cwd = Path.cwd()  
    if cwd.stem != dir_name:
        os.chdir(dir_name)

    # Run the shell script for HHblits

    # Get the sequences and write them to a fasta_file

    fasta_fname = f"{nproteins}_proteins.fasta" 
    for i, uid in enumerate(uid_list):
        if get_fastas_1:
            cmd_str = f"wget https://rest.uniprot.org/uniprotkb/{uid}.fasta" 
            subprocess.run(cmd_str, shell=True, check=True)
            subprocess.run(f"mv {uid}.fasta uniprot_seqs", shell=True, check=True)
        
        #cmd_str = f"cat {uid}.fasta >> uniprot_seqs/{fasta_fname}"
        #subprocess.run(cmd_str, shell=True)

        # Do an hhblits search

        if get_hhr_2:

            cmd_str = f"hhblits -i uniprot_seqs/{uid}.fasta " 
            cmd_str += f"-oa3m hhblits_out/{uid}.a3m " 
            cmd_str += f"-cpu {hhblits_cpu} "
            cmd_str += f"-qid {min_seq_id_with_query} "
            cmd_str += f"-cov {min_seq_coverage_with_query} "
            cmd_str += f"-e {min_e_value} "
            cmd_str += f"-d {pdb70_path} "
            cmd_str += f"> hhblits_out/{uid}.hhr"
    
            subprocess.run(cmd_str, shell=True, check=True)

            # output .hhr file

    if get_pdb_pair_ids_3:
        # Get the PDB ids with at least two potential
        # homologs
        pdb_ids = {}
        all_hits = []
        for file in Path("hhblits_out").iterdir():
            if file.suffix == ".hhr":
                file_str = str(file)
                hhr_dict: dict = HHBlits.parse_log(str(file))
                # parse the pdb file and it to the growing list
                uid = hhr_dict['uid']
                assert uid not in pdb_ids, uid

                hits = []

                potential_hits = hhr_dict["pdb_id"]
                assert len(potential_hits) > 0

                for i, prob in enumerate(hhr_dict["prob"]):
                    potential_homolog= False
                    prob = float(prob)
                    assert 0 <= prob <= 100
                    if prob >= 50:
                        potential_homolog = True
                    elif (prob >= 30) and (i < 3):
                        potential_homolog = True

                    if potential_homolog:
                        hit = potential_hits[i]
                        hits.append(hit)

                hits = list(set(hits))
                pdb_ids[uid] = hits
                all_hits = all_hits + hits

        
        all_hits = list(set(all_hits))
        all_hits = list(sorted(all_hits))
        if not Path("pdb_hits").is_dir():
            os.mkdir("pdb_hits")

        with open("pdb_hits/pdbs.json", "w") as f:
            json.dump(pdb_ids, f)

        with open("pdb_hits/all_hits", "w") as f:
            for pdb_id in all_hits:
                f.write(pdb_id + "\n")

        # count the pdb ids where at least two uniprot ids were found

        pdb_id__uids = {}
        for uid, pdb_id_list in pdb_ids.items():
            for pdb_id in pdb_id_list:
                if pdb_id not in pdb_id__uids:
                    pdb_id__uids[pdb_id] = [uid]
                else:
                    pdb_id__uids[pdb_id].append(uid)


        counts = {}
        for pdb_id, uids in pdb_id__uids.items():
            n = len(uids)
            assert n == len(set(uids))
            counts[pdb_id] = n

        with open("pdb_hits/counts.tsv", "w") as f:
            for pdb_id, count in counts.items():
                f.write(f"{pdb_id}    {count}\n")

        with open("pdb_hits/homologs.json", "w") as f:
            json.dump(pdb_id__uids, f)










        

    # Pairwise compare shared pdbs
