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

# ## CRL5 VIF CBF&#946; Benchmarking

# +
from collections import namedtuple
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

from src.cullin_benchmark_test import (
    CullinBenchMark,
    accumulate_indicies,
    bar_plot_df_summary,
    binary_search,
    biogrid_df_report,
    check_biogrid_data,
    check_bounds,
    compare_reports,
    find_bounds,
    find_end,
    find_start,
    format_biogrid_df_report,
    get_all_indicies,
    get_biogrid_summary,
    get_json_report_from_report,
    make_bounds,
    show_idmapping_results,
    transform_and_validate_biogrid,
    uniprot_id_mapping,
)

# +
# Global Notebook Flags
CHECK_BIOGRID_DATA = True
GET_BIOGRID_REPORT = True # Expensive
SAVE_BIOGRID_JSON = True
LOAD_BIOGRID_JSON = True
SHOW_BIOGRID_JSON = True

GET_BIOGRID_SUMMARY = True
PLOT_BIOGRID_SUMMARY = True
POST_UNIPROT_IDMAPPING = True
SHOW_UNIPROT_IDMAPPING = True
GET_CULLIN_BG = True
GET_CULLIN_BG_REPORT = True
SHOW_CULLIN_BG_REPORT = True

COMPARE_REPORTS = True

GET_CULLIN_BG_SUMMARY = True
PLOT_CULLIN_BG_SUMMARY = True



# +
cullin_benchmark = CullinBenchMark(dirpath=Path("../data/cullin_e3_ligase"))
#cullin_benchmark.load_data()

#cullin_benchmark.validate_prey()

cullin_benchmark.data.loc[cullin_benchmark.failed, "Prey"]

# + language="bash"
# shasum -c ../data/biogrid/checksum512.txt
# -

if CHECK_BIOGRID_DATA:
    check_biogrid_data()

# Load Biogrid into memory
biogrid = pd.read_csv("../data/biogrid/BIOGRID-ALL-4.4.206.tab3.txt", delimiter="\t")

biogrid = transform_and_validate_biogrid(biogrid)

# Long Running Cell
if GET_BIOGRID_REPORT:
    transformed_report = biogrid_df_report(biogrid)

# +
# Save the serializable output to json

if SAVE_BIOGRID_JSON:
    json_report = get_json_report_from_report(transformed_report)

    # Save the output of the previous cell
    with open("src/transformed_report.json", "w") as fp:
        json.dump(json_report, fp)

# -

LOAD_BIOGRID_JSON

if LOAD_BIOGRID_JSON:
    json_report = json.load(open("src/transformed_report.json", "r"))
if SHOW_BIOGRID_JSON:
    print(format_biogrid_df_report(**json_report))

# + language="bash"
# cat src/transformed_report.json
# -

if GET_BIOGRID_SUMMARY:
    summary = get_biogrid_summary(biogrid)
if PLOT_BIOGRID_SUMMARY:
    bar_plot_df_summary(biogrid, summary)

if POST_UNIPROT_IDMAPPING:
    id_mapping, failed_ids, prey_set_idmapping_input = uniprot_id_mapping(cullin_benchmark)
if SHOW_UNIPROT_IDMAPPING:
    show_idmapping_results(id_mapping, failed_ids, prey_set_idmapping_input)

# Prior Mappings  
# 13 failed to map  
# 2834 succeeded  
# of 2847 total  
#

# +
# Check the Failed Cases
failed_df = cullin_benchmark.data[cullin_benchmark.data['Prey'].apply(lambda x: x in failed_ids)]

# The failed cases amount to only 20 Bait prey Pairs
# The Saint Score < 0.14 for all cases
# Therefore we ignore the 13 failed cases instead of mapping them
# Except for L0R6Q1. Consider for later

# +
eids = set(val for key, val in id_mapping.items())

colA = "Entrez Gene Interactor A"
colB = "Entrez Gene Interactor B"

col = colA
eids_in_biogrid = set(map(lambda x: x if int(x) in biogrid.loc[:, col] else None, eids))
eids_in_biogrid.remove(None)
bounds_A = make_bounds(biogrid, col, eids_in_biogrid)

col = colB
eids_in_biogrid = set(map(lambda x: x if int(x) in biogrid.loc[:, col] else None, eids))
eids_in_biogrid.remove(None)
bounds_B = make_bounds(biogrid, col, eids_in_biogrid)
            

check_bounds(biogrid, bounds_A, eids, colnum=1)
check_bounds(biogrid, bounds_B, eids, colnum=2)

# +
if GET_CULLIN_BG:
    index_labels = get_all_indicies(biogrid, bounds_A, bounds_B, 1, 2)

    # Look at the subset of biogrid

    cullin_bg = biogrid.loc[index_labels]

    # Validate the data frame
    assert np.all(cullin_bg.iloc[:, 1].apply(lambda x: True if str(int(x)) in eids_in_biogrid else False))
    assert np.all(cullin_bg.iloc[:, 2].apply(lambda x: True if str(int(x)) in eids_in_biogrid else False))

    nnodes = len(eids_in_biogrid)
    n_possible_edges = int(0.5*nnodes*(nnodes-1))

    nself = len(cullin_bg[cullin_bg.iloc[:, 1]==cullin_bg.iloc[:, 2]])
    not_self = cullin_bg.iloc[:, 1]!=cullin_bg.iloc[:, 2]

    cullin_bg = cullin_bg[not_self]

    df = cullin_bg


    # How many edges are unique?

    


# +
if GET_CULLIN_BG_REPORT:
    cullin_report = biogrid_df_report(df)
    

# -

SHOW_CULLIN_BG_REPORT

# +
if SHOW_BIOGRID_JSON:
    print("Biogrid Report")
    print(format_biogrid_df_report(**json_report))
if SHOW_CULLIN_BG_REPORT:
    print("Cullin BG Report")
    print(format_biogrid_df_report(**cullin_report))
    
    
# -

if COMPARE_REPORTS:
    json_cullin_report = get_json_report_from_report(cullin_report)
    report_comparison = compare_reports(json_report, json_cullin_report, "Biogrid", "Cullin")

if GET_CULLIN_BG_SUMMARY:
    cullin_bg_summary = get_biogrid_summary(cullin_bg)
if PLOT_CULLIN_BG_SUMMARY:
    bar_plot_df_summary(cullin_bg, cullin_bg_summary)

report_comparison

# +
nodes = cullin_report['unique_GeneId_set']
d =pd.DataFrame(data = np.zeros((len(nodes), len(nodes))), index = nodes, columns = nodes, dtype=int)
print('{:,}'.format(d.values.nbytes))

def get_experimental_coverage_df(biogrid_df, report):
    
    df = biogrid_df
    nodes = report['unique_GeneId_set']
    nodes = sorted(nodes)
    d =pd.DataFrame(data = np.zeros((len(nodes), len(nodes))), index = nodes, columns = nodes, dtype=float)
    
    experiments = {key: i for i, key in enumerate(set(df["Experimental System"]))}
    
    j = 0
    for i, row in biogrid_df.iterrows():
        nodeA = row.iloc[1]
        nodeB = row.iloc[2]
        
        if nodeB > nodeA:
            t = nodeA
            nodeA = nodeB
            nodeB = t
        
        experiment = row["Experimental System"]
        val = experiments[experiment]
        
        d.loc[nodeA, nodeB] += val
        j +=1

    return d
        
        
        
        




# +
def split(u):
    a = u.split("|")
    for i in a:
        assert i.isdigit()
    a = [int(i) for i in a]
    for i in a:
        assert 0 <= i < 256
    a = np.array(a)
    return a

def to_entrez(u, id_mapping):
    return id_mapping[u] if u in id_mapping else "-"

def transform_cullin_benchmark_data(df, id_mapping):
    out = np.zeros((len(df), 4))
    out_ctrl = np.zeros((len(df), 12))
    entrez = []
    j=0
    for i, row in df.iterrows():
        out[j, :] = split(row["Spec"])
        out_ctrl[j, :] = split(row["ctrlCounts"])
        entrez.append(to_entrez(row["Prey"], id_mapping))
        j+=1
    
    
    for i in range(1, 5):
        #assert f"r{i}" not in df.columns
        df[f"r{i}"] = out[:, i-1]
        
    for i in range(1, 13):
        #assert f"ctrl_{i}" not in df.columns
        df[f"ctrl_{i}"] = out_ctrl[:, i-1]
        
    assert "Entrez" not in df.columns
    df["Entrez"] = entrez
    
    return df

    
# -

TRANSFORM_CULLIN = True
if TRANSFORM_CULLIN == True:
    cullin_benchmark_transformed = transform_cullin_benchmark_data(cullin_benchmark.data, id_mapping)

d = get_experimental_coverage_df(cullin_bg, cullin_report)


def cullin_benchmark_report(df):
    cbfb_sel = df["Bait"]=="CBFBwt_MG132"
    elob_sel = df["Bait"]=="ELOBwt_MG132"
    cul5_sel = df["Bait"]=="CUL5wt_MG132"
    
    def n_unique(sel, colname):
        return len(set(df.loc[sel, colname]))
    
    n_unique_CBFB_prey = n_unique(cbfb_sel, "Prey")
    n_unique_ELOB_prey = n_unique(elob_sel, "Prey")
    n_unique_CUL5_prey = n_unique(cul5_sel, "Prey")
    
    n_unique_CBFB_entrez = n_unique(cbfb_sel, "Entrez")
    n_unique_ELOB_entrez = n_unique(elob_sel, "Entrez")
    n_unique_CUL5_entrez = n_unique(cul5_sel, "Entrez")
    
    mid_confidence = df["SaintScore"] > 0.01
    
    
    
    return {'n_unique_CBFB_prey': n_unique_CBFB_prey,
           'n_unique_ELOB_prey': n_unique_ELOB_prey,
           'n_unique_CUL5_prey': n_unique_CUL5_prey,
           'n_unique_CBFB_entrez':n_unique_CBFB_entrez,
           'n_unique_ELOB_entrez':n_unique_ELOB_entrez,
           'n_unique_CUL5_entrez':n_unique_CUL5_entrez}

df = cullin_benchmark_transformed
cullin_benchmark_report(df)

df.apply(lambda x: len(set(x)))


# +
# Look at the Frequency of the Saint Scores
def plot_saint_score_frequency(df, l=8, w=20, title="Saint Score Frequency", bins=100,
                              rcParams={},
                               cmap = matplotlib.cm.tab10.colors
                              ):
    title += f"\nN={len(df)}"
    plt.figure(figsize=(w, l))
    plt.title(title)
    plt.ylabel("Frequency")
    plt.xlabel("Score")
    plt.hist(df["SaintScore"].values, bins=bins)
    plt.show()
    
plot_saint_score_frequency(cullin_benchmark.data)
# -

s1 = cullin_benchmark.data["SaintScore"] > 0.001
plot_saint_score_frequency(cullin_benchmark.data[s1])

# +
# Apply a set of filters
# SaintScore > 0.01
# 
df = cullin_benchmark_transformed
thresh = 0.01
s1 = df["SaintScore"] > thresh


s2 = df["Bait"] == "CBFBwt_MG132"
s3 = df["Bait"] == "ELOBwt_MG132"
s4 = df["Bait"] == "CUL5wt_MG132"

s5 = df["Entrez"] != "-"
a = s1 & s2
b = s1 & s3
c = s1 & s4


np.sum(s1), np.sum(s2), np.sum(s1 & s2), np.sum(s1 & s3), np.sum(s1 & s4), np.sum(s1 & s5)

# Select Saint score > thresh and Entrez ID not '-'
df = df[s1 & s5]
# -

df.apply(lambda x: len(set(x)))

# +
# Train on the first pulldown

s2 = df["Bait"] == "CBFBwt_MG132"
df = df[s2]
df.apply(lambda x: len(set(x)))

# +
#How many positives are there in the training set?

training_index_labels = sorted(df["Entrez"].apply(int))
colA = "Entrez Gene Interactor A"
s1 = cullin_bg[colA].apply(lambda x: int(x) in training_index_labels)
colB = "Entrez Gene Interactor B"
s2 = cullin_bg[colB].apply(lambda x: int(x) in training_index_labels)
s3 = s1 & s2
cullin_train_df = cullin_bg[s3]

# +
# Plot the Saint Scores for the training Data

plot_saint_score_frequency(df)
# -

TRANSFORM_CULLIN = True
if TRANSFORM_CULLIN == True:
    cullin_benchmark_transformed = transform_cullin_benchmark_data(cullin_benchmark.data, id_mapping)


def triangular_to_symmetric(A):
    Lower = np.tril(A)
    diag_indices = np.diag_indices(len(A))
    Lower = Lower + Lower.T
    Lower[diag_indices] = A[diag_indices]
    return Lower


def coverage_plot(d, w=12, h=12, title="Experimental Coverage of Cullin BG"):

    
    
    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(h)
    fig.set_figwidth(w)
    
    ax = axs[0]
    mappable = ax.imshow(triangular_to_symmetric(d.values != 0))
    ax.set_title(title)
    #plt.colorbar(mappable, ax=ax)
    ax = axs[1]
    ax.imshow(d.values)
    ax.set_title("Null")
    plt.tight_layout()
    plt.show()
coverage_plot(d, w=16, h=14,)

# +
cullin_train_df_report = biogrid_df_report(cullin_train_df)
d_train = get_experimental_coverage_df(cullin_train_df, cullin_train_df_report)

coverage_plot(d_train)
# -

print(format_biogrid_df_report(**cullin_train_df_report))


# +
# Dummy Model
def fit_dummy_model(key, nv, p, n_reps=None):
    if n_reps:
        M = np.zeros((nv, nv, n_reps))
        for i in range(n_reps):
            key, k1 = jax.random.split(key)
            rep = jax.random.bernoulli(k1, p=p, shape=(nv, nv))
            rep = np.array(rep)
            rep = triangular_to_symmetric(rep)
            M[:, :, i] = rep
    else:
        M = jax.random.bernoulli(key, p=p, shape=(nv, nv))
        M = np.array(M)
        M = triangular_to_symmetric(M)
    return M


nv = 268
n_reps = 100
M = fit_dummy_model(jax.random.PRNGKey(41), nv, 0.04, n_reps=n_reps)
M = np.array(M)

accuracies = np.zeros(n_reps)
precisions = np.zeros(n_reps)
if n_reps:
    y_true = d_train.values != 0
    y_true = triangular_to_symmetric(d_train.values)
    y_true = np.ravel(y_true)
    y_true = np.array(y_true, dtype=int)
    for rep in range(n_reps):
        mat = M[:, :, rep]
        accuracies[rep] = sklearn.metrics.accuracy_score(y_true=y_true, y_pred=np.ravel(mat))
        #precisions[rep] = sklearn.metrics.precision_score(y_true=y_true, y_pred=np.ravel(mat))
    
    
# Accuracy TP + FN / (TP + FP + TN + FN) Prec TP/(TP + F)


# -

bins=20
cmap = "bone"
fig, axs = plt.subplots(1, 2)
ax = axs[0]
ax.hist(accuracies, bins=bins)
ax.set_title("Accuracy")
ax = axs[1]
ax.set_title("Precision")
ax.hist(precisions, bins=bins)
plt.tight_layout()
plt.suptitle("Dummy Model")
print(np.sum(precisions))
plt.show()

plt.imshow(d_train != 0)

(d_train.values != 0)


# ?sklearn.metrics.accuracy_score

def plot_spectral_counts_distribution(df,
    columns = ["r1", "r2", "r3", "r4"],
    l=8,
    w=8,
    xlabel="Spectral Counts",
    ylabel="Frequency",
    title="Spectral Counts",
    bins=20,
    cmap=matplotlib.cm.tab10.colors,
    log10=False
):
    plt.figure(figsize=(w, l))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    
    if log10:
        data = np.log10(df[columns].values)
        plt.xlabel("log10 SC")
    else:
        data = df[columns].values
        plt.xlabel(xlabel)
    plt.hist(data, bins=20)
    plt.show()
    


transformed_cullin_df = transform_cullin_benchmark_data(cullin_benchmark.data)


plot_spectral_counts_distribution(transform_cullin_benchmark_data(cullin_benchmark.data), log10=False)
