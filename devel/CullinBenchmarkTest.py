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
        
        
        
        



# -

d = get_experimental_coverage_df(cullin_bg, cullin_report)


def triangular_to_symmetric(A):
    Lower = np.tril(A)
    diag_indices = np.diag_indices(len(A))
    Lower = Lower + Lower.T
    Lower[diag_indices] = A[diag_indices]
    return Lower


def coverage_plot(d, w=12, l=12, title="Experimental Coverage of Cullin BG"):
    w = 12
    l = w
    size = (l, w)
    plt.figure(figsize=size)
    plt.imshow(triangular_to_symmetric(d.values != 0))
    plt.title(title)
    plt.colorbar()
    plt.show()
coverage_plot(d)

np.sum(d.values != 0), np.min(d.values), np.max(d.values), d.values.shape

testdf = pd.DataFrame(index = np.arange(8), columns = np.arange(8), data=np.arange(64).reshape((8, 8)), dtype=float)

plt.imshow(testdf.values)
plt.colorbar()

# ?plt.imshow

testdf.

plt.imshow(np.arange(64).reshape((8, 8)))

# Imagine the biogrid data as an n x n x r tensor
n = len(set(cullin_bg.iloc[:, 1]))
r = len(set(cullin_bg.iloc[:, 3]))


np.sum(d.values != 0)


