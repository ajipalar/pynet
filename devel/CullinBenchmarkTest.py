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
import matplotlib.pyplot as plt
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
    
    
    def __init__(self, dirpath: Path):
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
        
        m = self.uniprot_acc_pattern(s)
        m = None if len(prey_str) in {0, 1, 2, 3, 4, 5, 7, 8, 9} else m
        m = None if len(prey_str) > 10 else m
        m = None if self.whitespace_pattern else m
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
                
        def get_unique_uniprot_prey(self):


cullin_benchmark = CullinBenchMark(dirpath=Path("../data/cullin_e3_ligase"))
cullin_benchmark.load_data()

cullin_benchmark.validate_prey()

cullin_benchmark.data.loc[cullin_benchmark.failed, "Prey"]
# -

type('abc') == str

type(type(None))

p = re.compile(r"[A-Z]")
m = p.match("S")

type(m)

# +
# Define the list of protein identifiers

cullin_benchmark.data
# -

prey_list = list(cullin_benchmark.data["Prey"].iloc[1:10])



prey_list


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

s

cullin_benchmark.data["Prey"].iloc[0:1000]

# + language="bash"
#
# -

help(requests.adapters)


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
# -



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


