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
    find_bounds,
    find_end,
    find_start,
    format_biogrid_df_report,
    get_all_indicies,
    get_biogrid_stats,
    get_biogrid_summary,
    make_bounds,
    show_biogrid_stats,
    show_idmapping_results,
    transform_and_validate_biogrid,
    uniprot_id_mapping,
)
# -

# Global Notebook Flags
CHECK_BIOGRID_DATA = True
GET_BIOGRID_REPORT = False
SAVE_BIOGRID_JSON = False
LOAD_BIOGRID_JSON = True
SHOW_BIOGRID_JSON = True
GET_BIOGRID_SUMMARY = True
PLOT_BIOGRID_SUMMARY = True
POST_UNIPROT_IDMAPPING = True
SHOW_UNIPROT_IDMAPPING = True
GET_CULLIN_BG_REPORT = True
SHOW_CULLIN_BG_REPORT = True

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

# Save the serializable output to json
if SAVE_BIOGRID_JSON:
    json_report = {key: int(transformed_report[key]) for key in ["n_interactions",
                                     "n_unique_GeneIds",
                                     "n_self_interactions",
                                     "n_non_self_interactions",
                                     "n_blank_GeneIds",
                                     "n_unique_edges",
                                     "n_blank_GeneIdsA",
                                     "n_blank_GeneIdsB",
                                     "n_blank_GeneIds"]}
    # Save the output of the previous cell
    with open("src/transformed_report.json", "w") as fp:
        json.dump(json_report, fp)


if LOAD_BIOGRID_JSON:
    json_report = json.load(open("src/transformed_report.json", "r"))
if SHOW_BIOGRID_JSON:
    print(format_biogrid_df_report(**json_report))

if GET_BIOGRID_STATS:
    stats = get_biogrid_stats(biogrid)
if SHOW_BIOGRID_STATS:
    show_biogrid_stats(stats)

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
            

check_bounds(biogrid, bounds_A, eids, colA, colnum=1)
check_bounds(biogrid, bounds_B, eids, colB, colnum=2)
# -

index_labels = get_all_indicies(biogrid, bounds_A, bounds_B, 1, 2)

# +
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

    

# -

if GET_CULLIN_BG_REPORT:
    cullin_report = biogrid_df_report(df)
if SHOW_BIOGRID_JSON:
    print("Biogrid Report")
    print(format_biogrid_df_report(**json_report))
if SHOW_CULLIN_BG_REPORT:
    print("Cullin BG Report")
    print(format_biogrid_df_report(**cullin_report))

# +
n_unique_edges = len(list(unique_edges))
density = n_unique_edges / n_possible_edges

s_nnodes = '{:,}'.format(nnodes)
s_n_possible_edges = '{:,}'.format(n_possible_edges)
s_n_interactions = '{:,}'.format(len(list(index_labels)))
s_n_self = '{:,}'.format(nself)
s_n_non_self = '{:,}'.format(len(df))
s_n_unique_edges = '{:,}'.format(n_unique_edges)
s_percent_density = np.round(density*100, decimals=4)

s = f"""For {s_nnodes} genes there are {s_n_possible_edges} possible interactions
{s_n_interactions} interactions were found in the biogrid
Of these interactions  {nself} are self interactions.
Leaving {s_n_non_self} non-self interactions.
{s_n_unique_edges} interactions are unique
The Cullin System BioGrid edge density is {s_percent_density}%
"""
print(s)
assert np.all(df.iloc[:, 1] != df.iloc[:, 2])

# +
# Get the frequencies of each experiment in the database
df = cullin_bg

experiment_keys = list(set(df.loc[:, esys]))
experiment_frequency = list(len(df[df.loc[:, esys] == key]) for key in experiment_keys)
experiment_type = list(next(iter(
    df.loc[df.loc[:, esys] == key, 
                etype])) for key in experiment_keys)


summary = pd.DataFrame({'Experiment': experiment_keys,
                        'Frequency': experiment_frequency,
                        'Type': experiment_type})
summary.sort_values('Frequency', inplace = True, ascending = False)
summary['log10(freq)'] = np.log10(summary['Frequency'])

# +
ax_params = {'x': np.linspace(0, 40, len(summary)),
             'title': f"Experimental Evidence Codes in Cullin Biogrid Subset (4.4.206)\
              \nN={len(df)}",
             "w": 12,
             "l": 8,
             "height": summary['log10(freq)'],
            }

cmap = matplotlib.cm.tab10.colors
colors = {'physical': cmap[0], 'genetic': cmap[1]}
x = np.linspace(0, 40, len(summary))

rcParams = {'font.size': 16}
fig, ax = plt.subplots(figsize=(ax_params['w'], ax_params['l']))
plt.rcParams.update(**rcParams)
plt.title(ax_params['title'])
plt.bar(x = ax_params['x'],
         height=summary['log10(freq)'],
         color=list(
             iter(map(lambda t: colors[t], summary['Type']
                     )
                 )
         ))
p_patch = mpatches.Patch(color=colors['physical'], label='physical')
g_patch = mpatches.Patch(color=colors['genetic'], label='genetic')

ax.set_xticks(x)
ax.set_xticklabels(labels=summary['Experiment'], rotation='vertical')
ax.set_ylabel("Log10 Frequency")
ax.legend(handles=[p_patch, g_patch])
plt.show()
# -

summary["frequency"] = 10 ** summary["log10(freq)"].values
summary = summary.sort_values("frequency", ascending=False)

q = 51009
lb, rb = bounds_A[q]
biogrid.iloc[lb:rb, :]

binary_search(4, [1, 2, 3], 0, 1, find_bounds)

str(q)

set(biogrid.iloc[[1,2, 7, 11, 99, 1234]].index)

biogrid.iloc[rb-1:lb+1, :]

biogrid[biogrid.loc[:, col]==q]

# +
col = "Entrez Gene Interactor A"

A_bounds = make_bounds(biogrid,)
# -

a = [0, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 5, 6, 7, 7, 88, 88, 9000]
b = [1]
c = [1, 2]
d = [1.1, 2.0, 2.01, 2.01, 2.01000, 3.14]
e = [0., 1., 2., 3., 3., 4., np.nan]

print(binary_search(np.float64(3), e, 0, len(e), find_bounds))

lb, rb = binary_search(1., biogrid.loc[:, col].values, 0, len(biogrid), find_bounds)





# +
def binary_search(n, a, start, end):
    """
    Find the entry
    """
    assert type(end) == int
    assert type(start) == int
    assert start <= end
    
    if start == end:
        return None
    
    middle = (start + end) // 2

    if a[middle] > n:
        return binary_search(n, a, start, middle)
    elif a[middle] < n:
        return binary_search(n, a, middle, end)
    else:
        return middle
    
def left_step(n, a, i, stepsize):
    if stepsize == 0:
        return i

    assert n == a[i]
    assert len(a) > 2
    while i - stepsize < 0:
        stepsize = stepsize // 2
    
    while i - stepsize > len(a) - 1:
        stepsize = stepsize // 2
        
        
    
        
    inclusive, exclusive = a[i-stepsize], a[i-stepsize]
    
        
    if n > a[i-stepsize]:
        # Take a smaller step
        return left_step(n, a, i, stepsize)
    elif n < a[i-stepsize]:
        
        
   


    
def get_left_index_in_block(n, a, i):
    """
    Args:
    
    Returns:
      i where i is the leftmost index
    
    """
    assert n == a[i]
    assert i < len(a) - 1
    
    
    step = len(a) / 100
    
    
    
    while n == a[i]:
        if i == 0:
            return i
        i -= 1
    return i + 1

def get_right_index(n, a, i):
    assert i < len(a)
    assert n==a[i]
        
    while n==a[i]:
        if i == len(a)-1:
            return i
        i += 1
    return i - 1
        
def get_block_bounds(n, a):
    loc = binary_search(n, a, 0, len(a))
    if loc == None:
        return tuple()
    return get_left_index(n, a, loc), get_right_index(n, a, loc)
