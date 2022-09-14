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

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import scipy as sp
import scipy.stats
import sys
from pathlib import Path
from functools import partial
import re
import itertools
import requests
import json


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
"""
Most of the Prey in the cullin E3 Ligase Benchmark are mapped with UniProt Accession Ids.
The biogrid interactors are identified using NCBI Entrez IDs


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


