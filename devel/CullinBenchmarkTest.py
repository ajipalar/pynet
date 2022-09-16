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
import re
import requests
import json
import scipy as sp
import scipy.stats
import sys


# +
# Load the system data into the notebook

class CullinBenchMark:
    
    def __init__(self, dirpath: Path, load_data=True, validate=True, create_unique_prey=True):
        self.path = dirpath
        self.data = pd.DataFrame()
        self.uniprot_re = r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}"
        self.baits = []
        self.prey = {}
        self.uniprot_acc_pattern = re.compile(self.uniprot_re)
        self.whitespace_pattern = re.compile(r".*\s.*")
        
        assert self.uniprot_acc_pattern.match("0PQRST") is None
        #assert p.match("A0B900C") is None
        assert self.uniprot_acc_pattern.match("a0b12c3") is None
        #assert p.match("O0AAA0AAA0") is None
        assert self.uniprot_acc_pattern.match("Q1AZL2") is not None
        
        if load_data:
            self.load_data()
            
        if validate:
            self.validate_prey()
            self.validate_spec()
            
        if create_unique_prey:
            self.create_unique_prey()
        
    def load_data(self):
        data_path = self.path / "1-s2.0-S1931312819302537-mmc2.xlsx"
        self.data = pd.read_excel(data_path)

        
    def unload_data(self):
        self.data = pd.DataFrame()
        self.failed = []
        self.baits = []
        self.prey = {}
        
    def get_protein_identifiers(self):
        ...
        
    def get_entrez_gid_from_uniprot_id(self):
        ...
        
    def get_gid_from_gene_name(self):
        ...
        
    def get_unidentified_proteins(self):
        ...
        
    def spec_counts_data(self):
        ...
        
    def reproduce_published_results(self):
        ...
        
    def validate_spec(self):
        for i, j in self.data.iterrows():
            assert len(j['Spec'].split('|')) == 4, print(j)
            assert len(j['ctrlCounts'].split('|'))==12, print(j)
            
    

    def validate_prey(self):
        """Checks prey are strings consistent with uniprot.
           puts the failed cases in a list"""
        # Regex consitent with Uniprot Accession
        
        failed = []
        r = 0
        for i, row in self.data.iterrows():
            
            prey_str = row["Prey"]
            m = self.is_uniprot_accesion_id(prey_str)
            if m is None:
                failed.append(i)
            r += 1
        assert r == len(self.data)
        self.failed = failed
        
    def is_uniprot_accesion_id(self, s: str) -> tuple[bool, None]:
        """Returns a re.Match object if s is consistent with
           a UniProt Accession identifier. Else None"""
        assert type(s) == str
        assert len(s) >= 0
        
        m = self.uniprot_acc_pattern.match(s)
        m = None if len(s) in {0, 1, 2, 3, 4, 5, 7, 8, 9} else m
        m = None if len(s) > 10 else m
        m = None if self.whitespace_pattern.match(s) else m
        return m
    
    

    def deprecated_validate_data_inner_loop(self, row):
        try:
            # uniport accession identifiers
            # see https://www.uniprot.org/help/accession_numbers
            assert 0 < len(row["Prey"]) <= 10
            assert len(row['Prey']) in (6, 10), row
            assert row['Prey'].isupper()
            assert row['Prey'].isalnum()
            assert row['Prey'][1].isdecimal()
            assert row['Prey'][5].isdecimal()
            try:
                assert row["Prey"][9].isdecimal()
                assert row["Prey"][6].isalpha()
            except IndexError:
                pass   
        except AssertionError:
            failed.append(i)
            
            
    def update_spec(self):
        
        spec = self.data['Spec']
        self.data = self.data.drop(columns='Spec')
          
    def create_unique_prey(self):
        self.unique_prey = set(self.data["Prey"])
        self.n_unique_prey = len(self.unique_prey)
        
    def __repr__(self):
        
        def f(exp, s):
            """Helper catcher wraps a try-except statement"""
            try:
                return s + exp()
            except AttributeError:
                return s
        
        s = ""
        s = f(lambda : f"shape : {self.data.shape}", s)
        #s = f(lambda : f"\ncolumns : {list(self.data.columns)}", s)
        s = f(lambda : f"\nnfailed : {len(self.failed)}", s)
        #s = f(lambda : f"\nfailed : {self.data.loc[cullin_benchmark.failed, 'Prey']}", s)

        return s



cullin_benchmark = CullinBenchMark(dirpath=Path("../data/cullin_e3_ligase"))
#cullin_benchmark.load_data()

#cullin_benchmark.validate_prey()

cullin_benchmark.data.loc[cullin_benchmark.failed, "Prey"]

# + language="bash"
# shasum -c ../data/biogrid/checksum512.txt
# -

# Check Biogrid Data
data_path = "../data/biogrid/BIOGRID-ALL-4.4.206.tab3.txt"
with open(data_path, 'r') as fi:
    nlines = len(fi.readline().split("\t"))
    assert nlines > 10
    j=1
    for i, line in enumerate(fi):
        le = len(line.split("\t"))
        assert le == nlines, f"{i, nlines, le}"
        j+=1
    print(f"n-lines {j}")

# Load Biogrid into memory
biogrid = pd.read_csv("../data/biogrid/BIOGRID-ALL-4.4.206.tab3.txt", delimiter="\t")
nbytes_0 = 0
for col in biogrid:
    nbytes_0 += biogrid.loc[:, col].nbytes

# +
col = "Entrez Gene Interactor A"
rows = biogrid.loc[:, col].apply(lambda x: type(x) == str)
#assert np.all(biogrid.loc[rows, col].apply(lambda x: x.isnumeric())) # Fails due to "-"
biogrid.loc[rows, col] = biogrid.loc[rows, col].apply(lambda x: np.nan if x=="-" else x)
rows = biogrid.loc[:, col].apply(lambda x: type(x) == str)
assert np.all(biogrid.loc[rows, col].apply(lambda x: x.isdigit())) # Passes
assert np.all(biogrid.loc[rows, col].apply(lambda x: x[0] != '0'))
biogrid.loc[rows, col] = biogrid.loc[rows, col].apply(lambda x: np.float64(x))
biogrid.loc[:, col]=pd.to_numeric(biogrid.loc[:, col])

col = "Entrez Gene Interactor B"
rows = biogrid.loc[:, col].apply(lambda x: type(x) == str)
#assert np.all(biogrid.loc[rows, col].apply(lambda x: x.isnumeric())) # Fails due to "-"
biogrid.loc[rows, col] = biogrid.loc[rows, col].apply(lambda x: np.nan if x=="-" else x)
rows = biogrid.loc[:, col].apply(lambda x: type(x) == str)
assert np.all(biogrid.loc[rows, col].apply(lambda x: x.isdigit())) # Passes
assert np.all(biogrid.loc[rows, col].apply(lambda x: x[0] != '0'))
biogrid.loc[rows, col] = biogrid.loc[rows, col].apply(lambda x: np.float64(x))
biogrid.loc[:, col]=pd.to_numeric(biogrid.loc[:, col])

col = "Score"
rows = biogrid.loc[:, col].apply(lambda x: type(x) == str)
biogrid.loc[rows, col] = biogrid.loc[rows, col].apply(lambda x: np.nan if x=="-" else x)
biogrid.loc[:, col]=pd.to_numeric(biogrid.loc[:, col])

categorical_cols = ["Experimental System", 
                    "Experimental System Type",
                    "Publication Source",
                    "Systematic Name Interactor A",
                    "Systematic Name Interactor B",
                    "Official Symbol Interactor A",
                    "Official Symbol Interactor B",
                    "Author",
                    "Publication Source",
                    "Organism ID Interactor A",
                    "Organism ID Interactor B",
                    "Synonyms Interactor A",
                    "Synonyms Interactor B",
                    "Throughput",
                    "Modification",
                    "Qualifications",
                    "Tags",
                    "Source Database",
                    "Ontology Term Categories",
                    "Ontology Term IDs",
                    "Ontology Term Names",
                    "Ontology Term Qualifier Names",
                    "Ontology Term Qualifier IDs",
                    "Ontology Term Types",
                    "SWISS-PROT Accessions Interactor A",
                    "SWISS-PROT Accessions Interactor B",
                    "TREMBL Accessions Interactor A",
                    "TREMBL Accessions Interactor B",
                    "REFSEQ Accessions Interactor A",
                    "REFSEQ Accessions Interactor B",
                    "Organism Name Interactor A",
                    "Organism Name Interactor B"
                   ]
for col in categorical_cols:
    biogrid.loc[:, col] = biogrid.loc[:, col].astype("category")


selected_cols = ['#BioGRID Interaction ID', 
                 'Entrez Gene Interactor A', 
                 'Entrez Gene Interactor B',
                 'Experimental System',
                 'Experimental System Type', 
                 'Score',
                 "Organism Name Interactor A",
                 "Organism Name Interactor B",
                 "Ontology Term Categories"
                 ]
biogrid = biogrid[selected_cols]
# -

nbytes = 0
for col in biogrid:
    nbytes += biogrid[col].nbytes

print(f"{np.log2(nbytes_0), np.log2(nbytes), biogrid.shape}")
print(f"{np.log10(nbytes_0), np.log10(nbytes)}")



# +

n_unique_genes = len(set(biogrid['Entrez Gene Interactor A']).union(set('Entrez Gene Interactor B')))
n_unique_interactions = len(set(biogrid['#BioGRID Interaction ID']))
n_possible_interactions = int(0.5 * n_unique_genes * (n_unique_genes -1))
interaction_density = n_unique_interactions / n_possible_interactions

print(f"Biogrid has {n_unique_genes} unique genes \
with {n_unique_interactions}.\nThe interaction density is {interaction_density}")
# -

etype = 'Experimental System Type'
esys = 'Experimental System'

# +
# Get the frequencies of each experiment in the database
experiment_keys = list(set(biogrid.loc[:, esys]))
experiment_frequency = list(len(biogrid[biogrid.loc[:, esys] == key]) for key in experiment_keys)
experiment_type = list(next(iter(
    biogrid.loc[biogrid.loc[:, esys] == key, 
                etype])) for key in experiment_keys)


summary = pd.DataFrame({'Experiment': experiment_keys,
                        'Frequency': experiment_frequency,
                        'Type': experiment_type})
summary.sort_values('Frequency', inplace = True, ascending = False)
summary['log10(freq)'] = np.log10(summary['Frequency'])

# +
ax_params = {'x': np.linspace(0, 40, len(summary)),
             'title': f"Experimental Evidence Codes in Biogrid (4.4.206)\
              \nN={len(biogrid)}",
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

# +
# Map IDS
col = "Prey"
rows = cullin_benchmark.data.loc[:, col].apply(
    lambda x: True if cullin_benchmark.uniprot_acc_pattern.match(x) else False)
unique_uniprot_prey = set(cullin_benchmark.data.loc[rows, col])

test_list = list(unique_uniprot_prey)[0:100]
data = {"from": "UniProtKB_AC-ID", "to": "GeneID", "ids":",".join(unique_uniprot_prey), "size": 5000}
API_URL = "https://rest.uniprot.org"
POST_END_POINT = f"{API_URL}/idmapping/run"
response = requests.post(POST_END_POINT, data=data)
assert response.status_code == 200
jobId = response.json()['jobId']

GET_END_POINT = f"{API_URL}/idmapping/status/{jobId}"
job_status = requests.get(GET_END_POINT)
assert job_status.status_code == 200
x_total_results = job_status.headers['x-total-results']
n_unique_mappable_prey = len(list(unique_uniprot_prey))


GET_END_POINT = f"{API_URL}/idmapping/stream/{jobId}"
stream_response = requests.get(GET_END_POINT)
assert job_status.status_code == 200
response_dict = stream_response.json()
response_dict = response_dict['results']
id_mapping = {}
for pair in response_dict:
    id_mapping[pair['from']] = pair['to']
    
failed_ids = job_status.json()['failedIds']
assert len(failed_ids) + len(set(id_mapping)) == len(unique_uniprot_prey)
# -

# Prior Mappings  
# 13 failed to map  
# 2834 succeeded  
# of 2847 total  
#

print(f"{len(failed_ids)} failed to map\n{len(set(id_mapping))} succeeded\nof {len(unique_uniprot_prey)} total")

# +
# Check the Failed Cases
failed_df = cullin_benchmark.data[cullin_benchmark.data['Prey'].apply(lambda x: x in failed_ids)]

# The failed cases amount to only 20 Bait prey Pairs
# The Saint Score < 0.14 for all cases
# Therefore we ignore the 13 failed cases instead of mapping them
# Except for L0R6Q1. Consider for later

# +
# How many interactions are there for the entrez genes

"""
Brute Force. Do not run
entrez_ids = [val for key, val in id_mapping.items()]


old_shape = biogrid.shape
col = "Entrez Gene Interactor A"
not_nan = biogrid.loc[:, col].apply(lambda x: not np.isnan(x))
biogrid = biogrid.loc[not_nan]
col = "Entrez Gene Interactor B"
not_nan = biogrid.loc[:, col].apply(lambda x: not np.isnan(x))
biogrid = biogrid.loc[not_nan]
# Filter Out the Places missing Entrez IDS
new_shape = biogrid.shape
col = "Entrez Gene Interactor A"

not_found_row_indicies = biogrid.index
for i, eid in enumerate(entrez_ids):
    rows = biogrid.loc[not_found_row_indicies, col].apply(lambda x: int(x) == int(eid))
    not_found_rows_indicies = rows[rows==False].index
    print((i, len(rows))) if i % 200 == 0 else ...


"""
# -


2 * 2834 * 2312698

"""
Search Problem Statement

For every gene in Cullin BenchMark, construct a subdata frame with those entries

Brute Force 2 * 2834 * 2,312,698 = 13, 108, 372, 264

find_indicies_from_sorted_list_A
  index set
find_indicies_from_sorted_list_B
  index set
Take the set_intersection

find_start
linear_search
"""


# +
def binary_search(n, a, start, end, linear):
    """
    Find the starting index i where a[i] == n
    If n not in a, return None
    """
    
    assert type(end) == int
    assert type(start) == int
    assert start <= end
    
    if start == end:
        return None
    
    middle = (start + end) // 2
    #print(start, middle, end)

    if a[middle] > n:
        return binary_search(n, a, start, middle, linear)
    elif a[middle] < n:
        return binary_search(n, a, middle+1, end, linear)
    else:
        """
        Do a linear search to the left to wind the start
        """
        return linear(n, a, middle, end)
    
def find_start(n, a, start, end):
    """
    Linear search for starting index
    """
    while a[start] == n:
        if start == 0:
            return start
        start -=1
    return start + 1

def find_end(n, a, start, end):
    """
    Linear search for ending index
    """
    assert start <= len(a)
    if start == len(a):
        return start
    while a[start] == n:
        start += 1
        if start == len(a):
            return start
    return start

def find_bounds(n, a, start, end):
    """
    Return the inclusive exclusive [lb, rb) left and right bounds of a contigous array
    whose values == n
    """
    assert 0 <= start <= end
    lb = find_start(n, a, start, end)
    rb = find_end(n, a, start, end)
    assert lb < rb, f"{start, end, lb, rb}"
    return lb, rb
    
def make_bounds(df, col, ids):
    df = df.sort_values(col, ascending=True)
    bounds = {}
    for eid in ids:
        eid = int(eid)
        val = binary_search(eid, df.loc[:, col].values, 0, len(df), find_bounds)
        if val:
            bounds[eid] = val
    return bounds


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

def check_bounds(df, bounds, ids, colnum):
    """
    Checks the assumptions of the bounds
    """
    
    assert np.nan not in bounds
    assert len(set(bounds)) <= len(ids)
    
    colname = df.columns[colnum]
    df = df.sort_values(colname, ascending=True)
    ldf = len(df)
    for key, bound in bounds.items():
        lb, rb = bound
        assert type(lb) == type(rb) == int
        assert 0 <= lb < rb <= ldf, f"{key, lb, rb, ldf}"
        
        if lb > 0:
            assert df.iloc[lb -1, colnum] < df.iloc[lb, colnum]
        if rb < len(df):
            assert df.iloc[rb-1, colnum] < df.iloc[rb, colnum], f""
            

check_bounds(biogrid, bounds_A, eids, colA, colnum=1)
check_bounds(biogrid, bounds_B, eids, colB, colnum=2)


# +
def accumulate_indicies(df, colnum, bounds):
    """
    Get the set of index labels from the dataframe
    Args:
    
    Params:
    
    
    """
    print(f"{len(bounds)} incoming bounds")
    colname = df.columns[colnum]
    array_indicies = set()
    df = df.sort_values(colname, ascending=True)
    for key, bound in bounds.items():
        lb, rb = bound
        assert 0 <= lb < rb < len(df)

        array_indicies = array_indicies.union(range(lb, rb))
        
    array_indicies = list(array_indicies)
    index_labels = set(df.iloc[array_indicies].index)
    return index_labels
    
def get_all_indicies(df, bounds_A, bounds_B, colnum_A, colnum_B):
    
    a_index_labels : set = accumulate_indicies(df, colnum_A, bounds_A)
    b_index_labels : set = accumulate_indicies(df, colnum_B, bounds_B)
    
    index_labels = a_index_labels.intersection(b_index_labels)
    return index_labels


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
def biogrid_df_report(df, colA="Entrez Gene Interactor A", colB="Entrez Gene Interactor B"):
    

    unique_GeneId_set = set()
    unique_GeneId_set = unique_GeneId_set.union(set(df.loc[:, colA]))
    unique_GeneId_set = unique_GeneId_set.union(set(df.loc[:, colB]))
    n_unique_GebeIds = len(unique_GeneId_set)
    
    unique_edges = set()
    
    not_identified_A = np.sum(np.isnan(df.loc[:, colA]))
    not_identified_B = np.sum(np.isnan(df.loc[:, colB]))
    not_identified_total = not_identified_A + not_identified_B
    
    for i, row in df.iterrows():
        e = row[colA], row[colB]
        unique_edges = unique_edges.union(frozenset(e))
        
    n_unique_edges = len(unique_edges)
    
    nself = len(df[df.iloc[:, 1]==df.iloc[:, 2]])
    
    
    
    
    

# +
n_cullin_bench = len(list(set(cullin_benchmark.data)))


s2 = f"""Biogrid:


The Biogrid 'Ground Truth' is a {cullin_bg.shape} database
  n unqiue nodes: {s_nnodes}
  n non-self interactions : {s_n_non_self}
  unique edges : {s_n_unique_edges}
  density : {s_percent_density}%
The Cullin AP-MS system is a {cullin_benchmark.data.shape} database
  n nodes : {}
  n mapped nodes : {}
  n failed mapped : {}
  n mapped to biogrid : {}
"""
print(s2)
# -



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


# -

i = 0
for j in biogrid.index:
    i+=1

i

binary_search(4, a, 0, len(a))

get_right_index(1, b, 0)

  

get_block_bounds(1, b)

search_right(3, a, 0)

np.sort(biogrid.loc[:, 'Entrez Gene Interactor A'])

biogrid = biogrid.sort_values(col, ascending=True)

biogrid

biogrid.sort_index()

biogrid.sort_values(col, ascending=True)



old_shape

len(biogrid.loc[not_found_rows, col])

cullin_benchmark[cullin_benchmark.data['Prey'].apply(lambda x: x in failed_ids)

# +
"""
Most of the Prey in the cullin E3 Ligase Benchmark are mapped with UniProt Accession Ids.
The biogrid interactors are identified using NCBI Entrez IDs

Map UniProt IDs to Entrez IDs

"""

    
    

# Update the Cullin E3 Ring Ligase with 9

# Remove failed cases


cullin_benchmark.data["Prey"]
# -

matplotlib.cm.tab10


# +
class UniProtIDMapping:
    """Maps UniProt Accession IDs to Entrez Gene Ids"""
    
    def __init__(self, ids, from_="UniProtKB_AC-ID", to="GeneID"):
        assert len(preys) < 100000, f"{len(preys)} preys over  idmapping limit"
        self.API_URL = "https://rest.uniprot.org"
        self.END_POINT = "idmapping/run"
        self.ids = ids
        self.request_data = {"from": from_,
                        "to": to,
                        "ids": ids,
                        "size": 500}
        self.preys = preys
        
        
        
    def post_job(self):
        self.r = requests.post(f"{self.API_URL}/{self.END_POINT}", data=self.request_data)
        self.jobId = self.r.json()['jobId']
        
    def update_job_status(self):
        self.job_status = requests.get(f"{self.API_URL}/idmapping/status/{self.jobId}")
        
    def get_results(self):
        return self.job_status.json()['results']
    
    def get_results_stream(self):
        self.stream = requests.get(f"{self.API_URL}/idmapping/stream/{self.jobId}")
        
def to_str(preys):
    s = ""
    for prey in preys:
        s += f"{prey},"
    s = s.strip(",")
    return s


# -

def get_prey_selector(db):
    """
    Remove the failed prey from the list
    """
    
    selector = []
    failed = db.failed
    failed = sorted(failed, reverse=True)
    x = None
    
    current_found = True
    unfinished = True
    
    for i in rdb.data.index:
        if current_found:
            current_found = False
            if len(failed) == 0:
                break
            
            current_failed = failed.pop()
            
        if i < current_failed:
            selector.append(i)
        elif i == current_failed:
            current_found = True
        else:
            assert False, f"{i, current_failed}
            
        
    return selector, failed


# +
class TesterIndex:
    index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    
class Testeroo:
    failed = [0, 1, 3, 7, 9]
    data = TesterIndex
    
get_prey_selector(Testeroo)
    
# -

preys = to_str(cullin_benchmark.data["Prey"])
#preys = "Q01196"
idmapping = UniProtIDMapping(ids = preys)
idmapping.post_job()
idmapping.get_job_status()
idmapping.get_results()
idmapping.job_status

idmapping.get_job_status()
idmapping.job_status.json()

idmapping.job_status.headers['x-total-results']

len(set(cullin_benchmark.data["Prey"]))

requests.get(idmapping.job_status.links['next']['url']).json()

idmapping.r.headers

idmapping.job_status.headers["Link"]


def to_str(preys):
    s = ""
    for prey in preys:
        s += f"{prey},"
    s = s.strip(",")
    return s


s =to_str(cullin_benchmark.data["Prey"].iloc[0:13])

cullin_benchmark.data["Prey"].iloc[0:1000]


# +
# Use the Gene ID as the unique identifier

# +
# Get the Spectral counts data for the system

# (bait_gene_id, prey_gene_id, spectral_counts)

# +
# Count all unique pairs # 132*131 / 2 = 8646 pairs

# +
# Query the protein databank for all structures with a pair

class CullinPDBQuery:
    
    def __init__(self):
        ...
        
    def query_pdb(self, gid_list):
        ...
        


# +
# Query BioGrid for Evidence of both pairs

# Have a UniProt Accession ID
# Need a Entrez-Gene Database ID

class CullinBioGridQuery:
    def __init__(self):
        self.path = Path("../data/biogrid/BIOGRID-ALL-4.4.206.tab3.txt")
        
    def query_pdb(self, gid_list):
        ...
        
    def load_biogrid_tab3(self):
        self.database = pd.read_csv(self.path, delimiter="\t")
        
        
        
biogrid_query = CullinBioGridQuery()
biogrid_query.load_biogrid_tab3()
# + language="bash"
# ls ../data/biogrid

# +
class BaseCoverage:
    def __init__(self):
        ...
        
    def plot_coverage(self):
        ...

class CullinPDBCoverage(BaseCoverage):
    def __init__(self):
        ...
        

class CullinBioGridCoverage(BaseCoverage):
    def __init__(self):
        ...
        
    def plot_coverage(self):
        ...
        
        
        


# +
# Define all the evidence codes 

# +
# Show the coverage of Experimental data
# -

# Example coverage plot
mat = np.random.rand(132*132).reshape((132, 132))
mat = np.tril(mat) + np.tril(mat).T - np.eye(132) * mat
plt.imshow(mat)

# +
# Genome Wide loss of Function screens?

# +
# Benchmarking cases
# For a given pair there is either no evidence or evidence
# PDB TPR (binary classification) (P)
# PDB FNR (binary classification) (P)
# PDB FPR (binary classifcation)  (N)
# Cannot know FPR
# -


