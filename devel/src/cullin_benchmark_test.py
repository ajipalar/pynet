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
import json
import re
import requests
import scipy as sp
import scipy.stats
import sys
import time


class CullinBenchMark:
    def __init__(
        self, dirpath: Path, load_data=True, validate=True, create_unique_prey=True
    ):
        self.path = dirpath
        self.data = pd.DataFrame()
        self.uniprot_re = (
            r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}"
        )
        self.baits = []
        self.prey = {}
        self.uniprot_acc_pattern = re.compile(self.uniprot_re)
        self.whitespace_pattern = re.compile(r".*\s.*")

        assert self.uniprot_acc_pattern.match("0PQRST") is None
        # assert p.match("A0B900C") is None
        assert self.uniprot_acc_pattern.match("a0b12c3") is None
        # assert p.match("O0AAA0AAA0") is None
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
            assert len(j["Spec"].split("|")) == 4, f"{j}"
            assert len(j["ctrlCounts"].split("|")) == 12, f"{j}"

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
            assert len(row["Prey"]) in (6, 10), row
            assert row["Prey"].isupper()
            assert row["Prey"].isalnum()
            assert row["Prey"][1].isdecimal()
            assert row["Prey"][5].isdecimal()
            try:
                assert row["Prey"][9].isdecimal()
                assert row["Prey"][6].isalpha()
            except IndexError:
                pass
        except AssertionError:
            failed.append(i)

    def update_spec(self):

        spec = self.data["Spec"]
        self.data = self.data.drop(columns="Spec")

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
        s = f(lambda: f"shape : {self.data.shape}", s)
        # s = f(lambda : f"\ncolumns : {list(self.data.columns)}", s)
        s = f(lambda: f"\nnfailed : {len(self.failed)}", s)
        # s = f(lambda : f"\nfailed : {self.data.loc[cullin_benchmark.failed, 'Prey']}", s)

        return s


def check_biogrid_data():
    print("Checking biogrid data for missing tabs")

    data_path = "../data/biogrid/BIOGRID-ALL-4.4.206.tab3.txt"
    with open(data_path, "r") as fi:
        nlines = len(fi.readline().split("\t"))
        assert nlines > 10
        j = 1
        for i, line in enumerate(fi):
            le = len(line.split("\t"))
            assert le == nlines, f"{i, nlines, le}"
            j += 1
        print("Passed")
        print(f"n-lines {j}")


def transform_and_validate_biogrid(df):
    """
    '-' characters are mapped to np.nan
    row a is checked as a string


    """

    def process(df, col):
        print(f"Processing {col}")
        rows = df.loc[:, col].apply(lambda x: type(x) == str)
        # assert np.all(df.loc[rows, col].apply(lambda x: x.isnumeric())) # Fails due to "-"
        df.loc[rows, col] = df.loc[rows, col].apply(lambda x: np.nan if x == "-" else x)
        rows = df.loc[:, col].apply(lambda x: type(x) == str)
        assert len(rows) > 2000000
        assert np.all(df.loc[rows, col].apply(lambda x: x.isdigit()))  # Passes
        assert np.all(df.loc[rows, col].apply(lambda x: x[0] != "0"))
        df.loc[rows, col] = df.loc[rows, col].apply(lambda x: np.float64(x))
        df.loc[:, col] = pd.to_numeric(df.loc[:, col])
        return df

    col = "Entrez Gene Interactor A"
    df = process(df, col)

    col = "Entrez Gene Interactor B"
    df = process(df, col)

    col = "Score"
    rows = df.loc[:, col].apply(lambda x: type(x) == str)
    df.loc[rows, col] = df.loc[rows, col].apply(lambda x: np.nan if x == "-" else x)
    df.loc[:, col] = pd.to_numeric(df.loc[:, col])

    categorical_cols = [
        "Experimental System",
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
        "Organism Name Interactor B",
    ]
    print("Change dtypes to categorical")
    for col in categorical_cols:
        df.loc[:, col] = df.loc[:, col].astype("category")

    print("Select columns")
    selected_cols = [
        "#BioGRID Interaction ID",
        "Entrez Gene Interactor A",
        "Entrez Gene Interactor B",
        "Experimental System",
        "Experimental System Type",
        "Score",
        "Organism Name Interactor A",
        "Organism Name Interactor B",
    ]
    df = df[selected_cols]
    return df



def get_biogrid_summary(df):
    """
    Gets a summary dataframe from a biogrid dataframe

    Summary:
      frequency
      Experiment
      Experiment Type

    """
    etype = "Experimental System Type"
    esys = "Experimental System"

    experiment_keys = list(set(df.loc[:, esys]))
    experiment_frequency = list(
        len(df[df.loc[:, esys] == key]) for key in experiment_keys
    )
    experiment_type = list(
        next(iter(df.loc[df.loc[:, esys] == key, etype])) for key in experiment_keys
    )

    summary = pd.DataFrame(
        {
            "Experiment": experiment_keys,
            "Frequency": experiment_frequency,
            "Type": experiment_type,
        }
    )
    summary.sort_values("Frequency", inplace=True, ascending=False)
    summary["log10(freq)"] = np.log10(summary["Frequency"])
    return summary


def bar_plot_df_summary(
    df,
    summary,
    w=12,
    l=8,
    x=None,
    N=None,
    height=None,
    title="Experimental Evidence Codes in Biogrid (4.4.206)",
    cmap=matplotlib.cm.tab10.colors,
    rcParams={"font.size": 16},
):

    if not x:
        x = np.linspace(0, 40, len(summary))

    if not N:
        N = len(df)

    if not height:
        height = summary["log10(freq)"]

    colors = {"physical": cmap[0], "genetic": cmap[1]}

    fig, ax = plt.subplots(figsize=(w, l))

    plt.rcParams.update(**rcParams)
    plt.title(title)
    plt.bar(
        x=x,
        height=summary["log10(freq)"],
        color=list(iter(map(lambda t: colors[t], summary["Type"]))),
    )
    p_patch = mpatches.Patch(color=colors["physical"], label="physical")
    g_patch = mpatches.Patch(color=colors["genetic"], label="genetic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels=summary["Experiment"], rotation="vertical")
    ax.set_ylabel("Log10 Frequency")
    ax.legend(handles=[p_patch, g_patch])
    plt.show()


def uniprot_id_mapping(cullin_benchmark, verbose=True, size=5000, waittime=15):
    if not verbose:
        def show(x):
            ...

    else:
        def show(x):
            print(x)

    col = "Prey"
    rows = cullin_benchmark.data.loc[:, col].apply(
        lambda x: True if cullin_benchmark.uniprot_acc_pattern.match(x) else False
    )
    prey_set_idmapping_input = set(cullin_benchmark.data.loc[rows, col])

    assert len(prey_set_idmapping_input) < size

    test_list = list(prey_set_idmapping_input)[0:100]

    fro = "UniProtKB_AC-ID"
    t = "GeneID"

    data = {"from": fro, "to": t, "ids": ",".join(prey_set_idmapping_input), "size": size}

    show(f"from {fro} to {t} size {size}")

    API_URL = "https://rest.uniprot.org"
    POST_END_POINT = f"{API_URL}/idmapping/run"

    response = requests.post(POST_END_POINT, data=data)
    assert response.status_code == 200
    jobId = response.json()["jobId"]
    show(
        f"""POST {POST_END_POINT}
          respone  {response.status_code}
          jobId {jobId}"""
    )
    show(f"Waiting {waittime} s")
    time.sleep(waittime)

    GET_END_POINT = f"{API_URL}/idmapping/status/{jobId}"
    job_status = requests.get(GET_END_POINT)
    assert job_status.status_code == 200
    x_total_results = job_status.headers["x-total-results"]
    n_unique_mappable_prey = len(list(prey_set_idmapping_input))
    show(f"""GET {GET_END_POINT}""")

    GET_END_POINT = f"{API_URL}/idmapping/stream/{jobId}"
    stream_response = requests.get(GET_END_POINT)
    show(f"""GET {GET_END_POINT}""")
    assert job_status.status_code == 200
    response_dict = stream_response.json()
    response_dict = response_dict["results"]
    id_mapping = {}
    for pair in response_dict:
        id_mapping[pair["from"]] = pair["to"]

    failed_ids = job_status.json()["failedIds"]
    assert len(failed_ids) + len(set(id_mapping)) == len(prey_set_idmapping_input)
    assert len(id_mapping) == len(set(id_mapping))
    return id_mapping, failed_ids, prey_set_idmapping_input

def show_idmapping_results(id_mapping, failed_ids, prey_set_idmapping_input):
    print(f"{len(failed_ids)} failed to map\n{len(id_mapping)} succeeded\nof {len(prey_set_idmapping_input)} total")

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

def biogrid_df_report(df, colA="Entrez Gene Interactor A", colB="Entrez Gene Interactor B",
                      verbose = True):
    """
    Gets a report of the dataframe

    An 'interaction' is a row in the biogrid database
    An 'edge' is a pair of different unordered gene ids

    Retruns:
      A dict containing
        n_interactions
        n_unique_GeneIds
        unique_GeneId_set
        n_self_interactions
        n_non_self_interactions
        n_blank_GeneIds 
        n_unique_edges
    """
    
    if verbose:
        def verbosity(ncalled):
            
            messages = [""]
            print(messages[ncalled])
            
            
            
            
    else:
        def verbosity(ncalled):
            return None
    
    
    
    
    n_interactions = len(df)
    unique_GeneId_set = set()
    unique_GeneId_set = unique_GeneId_set.union(set(df.loc[:, colA]))
    unique_GeneId_set = unique_GeneId_set.union(set(df.loc[:, colB]))
    n_unique_GeneIds = len(unique_GeneId_set)
    

    
    self_interaction_selector = df.iloc[:, 1]==df.iloc[:, 2]
    non_self_interaction_selector = df.iloc[:, 1]!=df.iloc[:, 2]
    n_self_interactions = np.sum(self_interaction_selector)
    n_non_self_interactions = np.sum(non_self_interaction_selector)
    
    assert n_self_interactions + n_non_self_interactions == len(df)
    
    
    n_blank_GeneIdsA = np.sum(np.isnan(df.loc[:, colA]))
    n_blank_GeneIdsB = np.sum(np.isnan(df.loc[:, colB]))
    n_blank_GeneIds = n_blank_GeneIdsA + n_blank_GeneIdsB
    
    unique_edges = dict()
    
    df = df[non_self_interaction_selector]
    
    unique_edge_labels = []

    N = len(df)
    
    for label, row in df.iterrows():
        e = row[colA], row[colB]
        unique_edges[frozenset(e)] = None
        i = int(label)
        if i % (N // 10) == 0: 
            print(f"{np.round((i/N)*100, decimals=2)}%")
        
    n_unique_edges = len(unique_edges)
    
    return {"n_interactions": n_interactions,
            "unique_GeneId_set": unique_GeneId_set,
            "n_unique_GeneIds": n_unique_GeneIds,
            "n_self_interactions": n_self_interactions,
            "n_non_self_interactions": n_non_self_interactions,
            "n_blank_GeneIds": n_blank_GeneIds,
            "n_unique_edges": n_unique_edges,
            "unique_edge_labels": unique_edge_labels,
            "unique_edges": unique_edges,
            "n_blank_GeneIdsA": n_blank_GeneIdsA,
            "n_blank_GeneIdsB": n_blank_GeneIdsB
            }

def format_biogrid_df_report(n_interactions,
                             n_unique_GeneIds,
                             n_self_interactions,
                             n_non_self_interactions,
                             n_blank_GeneIds,
                             n_blank_GeneIdsA,
                             n_blank_GeneIdsB,
                             n_unique_edges,
                             **kwargs) -> str:

    N_possible_edges = sp.special.comb(n_unique_GeneIds, 2, exact=True)
    percent_edge_density = (n_unique_edges/ N_possible_edges) * 100 
    percent_edge_density = np.round(percent_edge_density, decimals=4)

    def h(s):
        return '{:,}'.format(s)
    

    return f"""
N interactions {h(n_interactions)}
N self-interactions {h(n_self_interactions)}
N non-self interactions {h(n_non_self_interactions)}
N unique GeneIds {h(n_unique_GeneIds)}
N blank GeneIds A {h(n_blank_GeneIdsA)}
N blank GeneIds B {h(n_blank_GeneIdsB)}
N blank GeneIds {h(n_blank_GeneIds)}
N possible edges {h(N_possible_edges)}
N unique edges {h(n_unique_edges)}
Edge density {h(percent_edge_density)}%
"""

