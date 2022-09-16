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
