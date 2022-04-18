#!/usr/bin/env python
# coding: utf-8

from .typedefs import (
    Array,
    ColName,
    DataFrame,
    DeviceArray,
    Dimension,
    GeneID,
    Index,
    Matrix,
    PRNGKey,
    ProteinName,
    Series,
    State,
    Vector,
)

from .jittools import is_jittable
import Bio.PDB
from functools import partial
import graphviz
import inspect
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import mygene
import pandas as pd
from pathlib import Path
import scipy
from typing import Any, Callable, NewType, Union


def minmaxlen(col):
    """Use to determine the max, min length of a pipe
    delimeted entry in a dataframe"""
    max_l = 0
    min_l = 1e9
    for syn in col:
        synlen = len(syn.split("|"))
        max_l = synlen if synlen > max_l else max_l
        min_l = synlen if synlen < min_l else min_l
    return max_l, min_l


def load_biogrid_v4_4(dpath) -> DataFrame:
    """biogrid tab3 database to pandas dataframe"""
    # dpath = Path("../data/biogrid/BIOGRID-ALL-4.4.206.tab3.txt")
    biogrid_df: DataFrame = pd.read_csv(dpath, sep="\t")
    return biogrid_df


def col_map(
    mapf: Callable,
    df: DataFrame,
    col: ColName,
) -> DataFrame:
    """Maps the entries in a dataframe according to
    the mapping function mapf. Returns the newly
    mapped dataframe"""

    col_it = iter(df[col])
    col_it = map(mapf, col_it)
    df[col] = list(col_it)
    return df


def drop_columns(biogrid_df: DataFrame) -> DataFrame:
    """Drops certain columns from the biogrid pandas
    dataframe"""
    to_drop = [
        "BioGRID ID Interactor A",
        "BioGRID ID Interactor B",
        #'Systematic Name Interactor A',
        #'Systematic Name Interactor B',
        #'Official Symbol Interactor A',
        #'Official Symbol Interactor B',
        #'Synonyms Interactor A',
        #'Synonyms Interactor B',
        "Modification",
        "Throughput",
        "Score",
        "Qualifications",
        "Tags",
        "Source Database",
        "SWISS-PROT Accessions Interactor A",
        "TREMBL Accessions Interactor A",
        "REFSEQ Accessions Interactor A",
        "SWISS-PROT Accessions Interactor B",
        "TREMBL Accessions Interactor B",
        "REFSEQ Accessions Interactor B",
        "Ontology Term IDs",
        "Ontology Term Names",
        "Ontology Term Categories",
        "Ontology Term Qualifier IDs",
        "Ontology Term Qualifier Names",
        "Ontology Term Types",
    ]
    d = biogrid_df.drop(columns=to_drop, inplace=False)
    return d


def filter_missing_entrez_interactors(d) -> DataFrame:
    """Removes missing (-) entrez interactors
    from biogrid pandas dataframe"""
    d = d[d["Entrez Gene Interactor A"] != "-"]
    d = d[d["Entrez Gene Interactor B"] != "-"]
    return d


def get_frequency(col) -> dict[str, int]:
    """Counts the occurance of each categorical
    variable in a given pandas dataframe column.
    returns a dictionary of {"category" : count}"""

    freq_dict = {}
    for label in col:
        if label not in freq_dict:
            freq_dict[label] = 0
        else:
            freq_dict[label] += 1
    return freq_dict


def get_physical(d):
    """Built to filter the biogrid database to only
    physical interactions"""

    t = d["Experimental System Type"] == "physical"
    d = d[t]
    return d


def plot_physical_experiments(d):
    d = get_physical(d)
    plot_col(d, "Experimental System", topn=20)


def prepare_biogrid_fpipe():
    return [load_biogrid_v4_4, drop_columns, filter_missing_entrez_interactors]


def prepare_biogrid(dpath):
    """Preprocesses biogrid dataframe for downstream analysis"""

    d = load_biogrid_v4_4(dpath)
    d = drop_columns(d)
    d = filter_missing_entrez_interactors(d)

    upper = lambda i: str(i).upper()
    d = col_map(upper, d, "Systematic Name Interactor A")
    d = col_map(upper, d, "Systematic Name Interactor B")
    d = col_map(upper, d, "Official Symbol Interactor A")
    d = col_map(upper, d, "Official Symbol Interactor B")
    return d


def plot_col(d, colname, titlename=None, topn=12):
    """Plots a horizontal bar plot of the counts of each
    category in the column.

    d : pandas dataframe
    colname : str, column name
    titlename : str, name to use in plot title
    topn : int, plot the categories with the top n counts"""

    freq_dict = get_frequency(d[colname])
    freq_dict = dict(sorted(freq_dict.items(), key=lambda x: x[1], reverse=True))

    vals = list(freq_dict.values())[0:topn]
    keys = list(freq_dict.keys())[0:topn]

    # color = np.arange(len(keys))
    plt.barh(keys, vals)
    if titlename:
        plt.title(titlename)
    # plt.legend(keys)
    plt.show()


def load_tip49_spec_counts_dataset() -> DataFrame:
    """Loads in the human HEK293 tip49 dataset from the 2011
    Nature Methods SAINT paper, supplement 2."""

    data = Path("../data")
    apms = data / "apms"
    saint = (
        apms
        / "20110108_NATUREMETHODS_8_1_SAINTProbabilisticScoringOfAffinityPurification"
    )
    tip49 = saint / "tip49_supp2.A._Interactions.csv"
    tip49 = pd.read_csv(tip49)
    tip49 = tip49.rename(columns=tip49.iloc[1])
    tip49 = tip49.iloc[2:, 1:]

    # print(tip49.columns[1])
    assert tip49.columns[1] == "Prey"
    assert tip49.columns[0] == "Bait"

    upper = lambda i: i.upper()
    tip49 = col_map(upper, tip49, "Bait")
    tip49 = col_map(upper, tip49, "Prey")
    return tip49


def print_baitprey_info(spec_counts_df):
    """prints some information on baits and preys
    does an assertion check
    For tip49
    Baits: 27
    Preys: 1207"""

    nbaits = len(set(spec_counts_df["Bait"]))
    print(f"Baits: {nbaits}")

    npreys = len(set(spec_counts_df["Prey"]))
    print(f"Preys: {npreys}")
    assert nbaits == 27
    assert npreys == 1207


def check_if_tip49_proteins_in_biogrid(biogrid: DataFrame, spec: DataFrame):
    """Checks the biogrid dataframe for bait and prey proteins in tip49
    so that interaction may be mapped"""

    def p(x, y) -> tuple[dict]:
        print(f"{x} {y}")

    sapiens = "Homo sapiens"
    sel = biogrid["Organism Name Interactor A"] == sapiens
    biogrid = biogrid[sel]
    p("dhuman", biogrid.shape)
    sel = biogrid["Organism Name Interactor B"] == sapiens
    biogrid = biogrid[sel]
    p("dhuman", biogrid.shape)
    off_sym_int_a_col = set(biogrid["Official Symbol Interactor A"])
    off_sym_int_b_col = set(biogrid["Official Symbol Interactor B"])
    sys_name_int_a_col = set(biogrid["Systematic Name Interactor A"])
    sys_name_int_b_col = set(biogrid["Systematic Name Interactor B"])

    baits = set(spec["Bait"])
    preys = set(spec["Prey"])

    def search_biogrid(
        query: set[ProteinName], db: DataFrame
    ) -> dict[ProteinName, list[ColName]]:

        """searchs the following biogrid columns for the bait
        or prey protein.

        Official Symbol Interactor A
        Official Symbol Interactor B
        Systematic Name Interactor A
        Systematic Name Interactor B

        TODO: Parse and search the synonyms column"""

        query_dict = {}

        query_len = len(query)
        for i, q in enumerate(query):
            print(f"{i}/{query_len} searching {q}")
            ql = []
            if q in off_sym_int_a_col:
                ql.append("Official Symbol Interactor A")
            if q in off_sym_int_b_col:
                ql.append("Official Symbol Interactor B")
            if q in sys_name_int_a_col:
                ql.append("Systematic Name Interactor A")
            if q in sys_name_int_b_col:
                ql.append("Systematic Name Interactor B")

            query_dict[q] = ql
        return query_dict

    baits_found: dict = search_biogrid(baits, biogrid)
    preys_found: dict = search_biogrid(preys, biogrid)
    return baits_found, preys_found


def is_protein_name_in_biogrid(query_dict: dict[ProteinName, list[ColName]]):
    """Checks if a protein name was found in biogrid

    query_dict : {[protein_name : [column_name]}

    The colnames represent the columns that the protein_name is in"""

    found_dict = {}
    for i, (key, val) in enumerate(query_dict.items()):
        found = True if len(val) > 0 else False
        found_dict[key] = found
    return found_dict


def is_protein_name_not_in_biogrid(query_dict) -> list[ProteinName]:
    """Checks if a protein name was not found in biogrid dataframe
    returns a list of all proteins not found in biogrid

    The query dict is the ouptput of the function
    `check_if_tip49_proteins_in_biogrid`"""
    lost_proteins = []
    for i, (pname, col_list) in enumerate(query_dict.items()):
        if len(col_list) == 0:
            lost_proteins.append(pname)
    return lost_proteins


def fraction_found(found: dict[ProteinName, bool]) -> tuple[float, int]:
    """Calculates the fraction fractions of baits or preys
    that were found in biogrid database.

    found : output of is_protein_name_in_biogrid
    returns frac, nkeys

    frac : float = nfound / nkeys
    nkeys : number of baits or preys"""

    nkeys = 0
    nfound = 0
    for i, (key, val) in enumerate(found.items()):
        nkeys += 1
        if val:
            nfound += 1
    return nfound / nkeys, nkeys


def pipe_to_gvis(fpipe: list[Callable]):
    """Returns a graphviz.Digraph object representing
    a linear functional pipe.

    fpipe : an ordered list of pure functions

    returns control_flow

    control_flow : graphviz.Digraph object
    where graphviz is python-graphviz
    """

    control_flow = graphviz.Digraph()
    for i, func in enumerate(fpipe):
        name = str(i)
        label = func.__name__
        control_flow.node(name, label)

    for i in range(len(fpipe) - 1):
        anno = fpipe[i].__annotations__
        r_t_str = str(anno["return"])
        control_flow.edge(str(i), str(i + 1), label=r_t_str)

    return control_flow


def find_idmapping_overlap(biogrid, spec_counts_df):
    """Wrapper functions that finds the overlap amongs bait and prey
    names in the tip49 dataframe and the biogrid dataframe"""

    baits_found, preys_found = check_if_tip49_proteins_in_biogrid(
        biogrid, spec_counts_df
    )
    baits_found = is_protein_name_in_biogrid(baits_found)
    preys_found = is_protein_name_in_biogrid(preys_found)
    baits_per, baits_n = fraction_found(baits_found)
    preys_per, preys_n = fraction_found(preys_found)

    print(f"% baits found {baits_per} of {baits_n}")
    print(f"% preys found {preys_per} of {preys_n}")


def assert_valid_pure_functional_pipe(fpipe):
    """Pure functions do not modify the global state of the
    program, the only operate on their inputs and return outputs.

    This function checks that the return type signature
    in the previous function matches the paramter signature in the
    current function

    fpipe is a list of pure functions to be sequentially applied
    as in f | g | h | i | j |k where f, g, h, i, j, and k are pure
    functions.

    Not yet working"""

    def get_params_and_return(functotest):
        annotations = functotest.__annotations__
        r_t = annotations.pop("return")
        p_t = annotations
        return p_t, r_t

    p_t, r_t = get_params_and_return(fpipe[0])
    for i in range(1, len(fpipe)):
        curr_p_t, curr_r_t = get_params_and_return(fpipe[i])

        try:
            assert curr_p_t == r_t
        except AssertionError:
            print(fpipe[i].__name__, curr_p_t, r_t)

        p_t = curr_p_t
        r_t = curr_r_t


def filter_biogrid(tip49: DataFrame, biogrid: DataFrame):
    """Filter out all biogrid entries that do not
    contain candidate proteins from the tip49 dataset"""

    # numpy matrix of biogrid columns
    m = biogrid[
        [
            "Systematic Name Interactor A",
            "Systematic Name Interactor B",
            "Official Symbol Interactor A",
            "Official Symbol Interactor B",
            "Synonyms Interactor A",
            "Synonyms Interactor B",
        ]
    ].values

    rownames = biogrid.index.values
    tip49names = set(tip49["Bait"]).union(tip49["Prey"])

    rownames_to_drop = []
    for i, row in enumerate(m):
        synonyms = row[4].split("|") + row[5].split("|")
        synonyms = list(map(lambda i: i.upper(), iter(synonyms)))
        names = list(row[0:4]) + synonyms
        names = set(names)

        intersection = tip49names.intersection(names)
        if len(intersection) == 0:
            rownames_to_drop.append(rownames[i])

        if i % 100000 == 0:
            print("=", end="")

    biogrid = biogrid.drop(index=rownames_to_drop)
    return biogrid


def jbuild(f, **partial_kwargs) -> Callable:
    """jit compile a function with static *args and **kwargs
    params:
      partial_kwargs : kwargs for the functools partial function
    return:
      jf : a jit complied function"""

    pf = partial(f, **partial_kwargs)
    jf = jax.jit(pf)
    return jf


def _get_vec_minus_s(s: Index, vec: Vector, vec_l: int) -> Vector:  # base 1 index
    """Returns a vec_l - 1 length vector equal to
    vec with the ith element removed"""

    a = np.zeros(vec_l - 1, dtype=vec.dtype)

    a = jax.lax.fori_loop(0, s - 1, lambda j, a: a.at[j].set(vec[j]), a)
    a = jax.lax.fori_loop(s, vec_l, lambda j, a: a.at[j - 1].set(vec[j]), a)
    return a


def get_matrix_col_minus_s(
    s: Index, phi: Matrix, p: Dimension  # base 1 index
) -> Vector:
    """Returns the ith column of the matrix phi with the ith element removed
    to jit compile. O(p)
      pf = parital(f, p=literal_p)
      jf = jax.jit(pf)
      or jbuild(f, **{'p':p})

    """
    # a = jnp.zeros(p-1, dtype=phi.dtype)
    col: Vector = phi[:, s - 1]  # a p length vector
    # a = jax.lax.fori_loop(0, i, lambda j, a: a.at[j].set(col[j]), a)
    # a = jax.lax.fori_loop(i+1, p, lambda j, a: a.at[j-1].set(col[j]), a)

    a: Vector = _get_vec_minus_s(s, col, p)  # a p-1 length vector
    return a


def get_random_phi_matrix(key: PRNGKey, p: Dimension) -> Matrix:
    """Generates the matrix phi for use in the
    Poisson Square Root Graphical Model Inouye 2016
    params:
      key : jax.random.PRNGKey
      p   : Dimension
      lam :
    return:
      phi : p x p matrix whose entries are sampled from independant
            univariate poisson distributions with rate lam"""

    phi = jax.random.normal(key, (p, p))
    phi_diag = phi.diagonal()
    phi = jnp.tril(phi) + jnp.tril(phi).T
    phi, diag = set_diag(phi, phi_diag)
    return phi


def get_random_X_matrix(key: PRNGKey, p: Dimension, n: Dimension, lam: float) -> Matrix:

    return jax.random.poisson(key, lam, (p, n))


def set_diag(phi: Matrix, diag: Vector) -> tuple[Matrix, Vector]:
    """Set the diagonal elements of the DeviceArray phi"""
    p = len(phi)
    phi_diag = jax.lax.fori_loop(
        0, p, lambda i, t: (t[0].at[(i, i)].set(t[1][i]), t[1]), (phi, diag)
    )

    return phi_diag


def get_eta1(phi: Matrix, s: Index) -> float:  # R^{p x p}  # base 1 index
    """
    params:
      theta : length p vector
      phi   : p x p matrix
      x_s   : p length vector
        s   : index from 1 to p
        i   : index from 1 to n
    return:
      eta1 : float
    """

    eta1 = phi[s - 1, s - 1]
    return eta1


def _get_eta2(
    theta_s: float,  # theta at index s-1
    phi_minus_s: Vector,  # a p - 1 length vector
    x_minus_s: Vector,  # a p-1 length data vector
    s: Index,  # base 1 index
    p: Dimension,
) -> float:
    """
    params:
      theta : length p vector
      phi   : p x p matrix
      x_s   : p length vector
        s   : index from 1 to p
        i   : index from 1 to
        p   : the problem dimensionality
    return:
      eta2 : float

    """

    eta2 = theta_s + 2 * phi_minus_s @ jnp.sqrt(x_minus_s)
    return eta2


def get_eta2(theta: Vector, phi: Matrix, x_i: Vector, s: Index, p: Dimension) -> float:

    col_s = phi[:, s - 1]
    theta_s = theta[s - 1]
    phi_minus_s = _get_vec_minus_s(s, col_s, p)
    x_minus_s = _get_vec_minus_s(s, x_i, p)
    eta2 = _get_eta2(theta_s, phi_minus_s, x_minus_s, s, p)
    return eta2


def logsum(x):
    """Sum i=1 to x log(i)"""
    return jax.lax.fori_loop(1, x + 1, lambda i, val: val + jnp.log(i), 0)


def logfactorial(x_si):
    """Jit compileable log factorial log(n!) = Sum i=1 to n log(n)
    if n == 0 return log(0!) = log(1) = 0
    if n == 1 return log(1!) = log(1) = 0

    """
    return jax.lax.cond(x_si <= 1, lambda x: 0.0, logsum, x_si)


def Zexp(eta1, eta2):
    """Not jittable because we use the scipy erfc implementation
    eta1 < 0 by 13
    sqrt(pi)*e^{-eta2^2/4eta1}(1 - erf(-eta2/2*sqrt(-eta1)))/-2*(-eta1)^{3/2} - 1/eta1
    """
    real1 = jnp.sqrt(jnp.pi) * eta1 * jnp.exp(-(eta2 ** 2) / (4 * eta1))

    c1 = complex(eta1)
    erf_arg = -eta2 / (2 * jnp.sqrt(c1))
    complex1 = scipy.special.erfc(erf_arg)
    denominator = -2 * (jnp.sqrt(c1) ** 3)
    subtract = 1 / eta1

    # Assumption, return the real component

    return jnp.real(real1 * complex1 / denominator - subtract)


def get_exp_random_phi(key, p, unimin=-1.0, unimax=-1e-05):
    """gamma = 0 thus phi_exp is 0 on the off diagonal elements

    params:
      key
      p
      unimin : min val for eta1
      unimax : max val for eta1. Note eta < 0
    return:
      phi_exp : a p x p matrix. Note phi is negative definite
    """

    key, k1, k2 = jax.random.split(key, 3)
    diag = jax.random.uniform(k1, shape=(p,), minval=unimin, maxval=unimax)
    phi_exp = jnp.zeros((p, p))
    phi_exp, diag = set_diag(phi_exp, diag)
    return phi_exp


def Aexp(eta1, eta2):
    return jnp.log(Zexp(eta1, eta2))


def f0(xsi, eta1, eta2):
    return jnp.exp(eta1 * xsi + eta2 * jnp.sqrt(xsi) - logfactorial(xsi))


def fn(xsi, eta1):
    return jnp.exp(eta1 * xsi)


# In[58]:


def test_matrix_minus_slice(slicef):
    seed = 7
    key = jax.random.PRNGKey(seed)
    p = 5
    poisson_lam = 8
    phi = get_poisson_matrix(key, p, poisson_lam)

    jslice_neg_s = jbuild(slicef, **{"p": p})

    for i in range(13):
        print(i)
        col = phi[:, i]
        sl = jslice_neg_s(i, phi)
        # print(i, col, sl)
        if i == 0:
            # print(i, col, sl)
            assert jnp.all(col[1:p] == sl)
        elif i < len(phi):
            assert jnp.all(col[0:i] == sl[0:i])
            assert jnp.all(col[i + 1 : p] == sl[i : len(sl)])
        else:
            assert jnp.all(col[0 : p - 1] == sl)


def get_ncbi_gene_id(name, mg) -> GeneID:
    """
    params:
        name :
        mg : mygene.MyGeneInfo() object
    return :
        An NCBI gene id -> int or str?
    """

    return mg.query(name)["hits"][0]["_id"]


def get_mmcif_structure(name: str, dpath: Path) -> object:
    """Devnote: Nothing consumes this function
    get a structure parser object from an mmcif data path.

    params:
       name : a non-unique identifier ascociated with the structure
       dpath: the pathlib.Path to the mmcif file containing the structure

    return:
       structure : a Bio.PDB.Structure.Structure object
    """

    parser = Bio.PDB.MMCIFParser()
    structure = parser.get_structure(name, str(dpath))
    return structure


def get_saga_mmcif_structure() -> object:
    """Gets the saga complex for benchmarking the Poisson SQR Model

    return : a Bio.PDB.Structure.Structure object
    """

    saga_path = Path("../data/saga_complex/7KTR.cif")
    structure = get_mmcif_structure("SAGA", saga_path)
    return structure


def dev_surface_plot_block():

    from mpl_toolkits.mplot3d import Axes3D

    # Axes3D import has side effects, it enables using projection='3d' in add_subplot
    import random

    def fun(x, y):
        return -(x ** 2) - y ** 2

    fs = 8
    fig = plt.figure(figsize=(fs, fs))
    ax = fig.add_subplot(111, projection="3d")
    x = y = np.arange(-3.0, 3.0, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array(fun(np.ravel(X), np.ravel(Y)))
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    plt.tight_layout()

    plt.show()


def dev_numerical_approximation_block():
    # Numerical Approximation

    # integral 1/5 dx/x

    def f(x):
        return 1 / x

    n = 50
    a = 1
    b = 5
    deltax = (b - a) / n

    def xk(k, n):
        return 1 + 4 * k / n

    xkarr = np.array(list(xk(k, n) for k in iter(range(0, n + 1))))
    Mn = sum(list(f((xkarr[i] + xkarr[i - 1]) / 2) * deltax for i in range(1, n + 1)))
    Tn = sum(
        list((f(xkarr[i]) + f(xkarr[i - 1])) / 2 * deltax for i in range(1, n + 1))
    )
    S2n = 2 / 3 * Mn + 1 / 3 * Tn

    # In[ ]:

    def integrate(dx):
        x = np.arange(1, 5, dx)
        y = 1 / x
        return sum(dx * y)

    # In[ ]:

    dxs = np.exp(range(0, -13, -1))
    y = list(integrate(dx) for dx in dxs)
    plt.plot(np.log(dxs), y)

    # The numerical approximation converages at the value of the integral

    # In[ ]:

    dMn = list(Mn - i for i in y)
    dTn = list(Tn - i for i in y)
    dS2n = list(S2n - i for i in y)

    plt.plot(dxs, dMn)
    plt.plot(dxs, dTn)
    plt.plot(dxs, dS2n)
    plt.ylim()

    # In[ ]:

    import math

    erf = math.erf
    jerf = jax.jit(erf)

    # In[ ]:

    # In[ ]:

    import inspect

    # In[ ]:

    z = np.arange(0.1, 10, 0.1)
    x = map(lambda i: math.gamma(i), iter(z))
    x = list(x)
    y = x
    x = np.arange(0.1, 10, 0.1)
    plt.plot(x, y)

    # In[ ]:

    x = map(lambda i: math.factorial, iter(np.arange()))


def dev_ncbi_block():
    """From ipynb block for development"""
    import requests

    def pyiter_to_slist(pyiter: list) -> str:
        s = ""
        for i in pyiter:
            s += str(i) + ","
        s = s.strip(",")
        return s

    def post_ncbi(genelist):
        q = pyiter_to_slist(genelist)
        headers = {"content-type": "application/x-www-form-urlencoded"}
        params = f"q={q}&scopes=symbol&fields=_id&species=human"
        res = requests.post("http://mygene.info/v3/query", data=params, headers=headers)
        return res

    POSTs = NewType("POSTs", str)

    def get_geneid_from_respone(response: POSTs) -> list[GeneID]:
        gene_ids: list[GeneID] = []
        for i, d in enumerate(response.json()):
            try:
                gene_ids.append(d["_id"])
            except KeyError:
                print(d)
        return gene_ids

    def get_ncbi_gene_id(name):

        return mg.query(name).json()["hits"][0]["_id"]

    def test_post_ncbi(biogrid):

        test_case = biogrid["Official Symbol Interactor A"]
        q = list(test_case)
        q = pyiter_to_slist(test_case)
        q = test_case

        res = post_ncbi(q)
        gene_ids = get_geneid_from_respone(res)

        for i, (rname, row) in enumerate(biogrid.iterrows()):
            if i % 100 == 0:
                print("=", end="")
            gene_id = str(gene_ids[i])
            biogrid_id = str(row["Entrez Gene Interactor A"])
            assert gene_id == biogrid_id

    # In[ ]:

    test_post_ncbi(biogrid.iloc[0:1000])

    mg.query("MRE11A", species="human")

    genelist = list(set(spec_counts_df["Bait"]))
    res = post_ncbi(genelist)

    list(biogrid["Official Symbol Interactor A"])

    pyiter_to_slist(biogrid["Official Symbol Interactor A"].iloc[0:20])

    for i in biogrid["Official Symbol Interactor A"].iloc[0:20]:
        print(i)


def dev_roc_examples_block():
    """dev block, roc prc curves"""
    # ROC Examples
    key = jax.random.PRNGKey(10)
    y_true = jax.random.bernoulli(key, p=0.1, shape=(len(spec_counts_df),))
    y_true = np.array(y_true, dtype=int)

    y_score = spec_counts_df["SAINT"].values
    y_score = map(lambda i: float(i), iter(y_score))
    y_score = list(y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    prec, recall, pthresh = precision_recall_curve(y_true, y_score)

    plt.subplot(121)
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.plot(fpr, tpr)
    plt.title("ROC")
    plt.subplot(122)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.plot(recall, prec)
    plt.title("PRC")
    plt.tight_layout()
    plt.show()


def dev_get_dev_state_poisson_sqr():
    seed = 7
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, num=3)
    p: Dimension = 5
    n = 3
    lam: float = 9.1
    X = get_random_X_matrix(k1, p, n, lam)
    phi = get_random_phi_matrix(k2, p)
    theta = jax.random.normal(k3, shape=(p,))

    return theta, phi, X, p, n


"""

def dev_is_jittable(theta : Vector, 
        phi : Matrix, 
        x_minus_s : Vector, # p-1 length vector
        x : Vector, # a p length vector
        s : Index, 
        i : Index,
        p : Dimension):

    theta_s = theta[s - 1]

    test_cases = {'get_eta1': get_eta1,
            'get_eta2': get_eta2,
            'get_poisson_matrix': get_poisson_matrix,
            'logfactorial': logfactorial,
            'logsum': logsum,
            'f0': f0,
            'fn': fn,
            'get_matrix_col_minus_s': get_matrix_col_minus_s,
            '_get_vec_minus_s': _get_vec_minus_s
            }

    args = {'get_eta1': (phi, s),
            'get_eta2': (theta_s, phi, x_minus_s, s, p), 
            'get_poisson_matrix': ,
            'logfactorial': ,
            'logsum': ,
            'f0': ,
            'fn': ,
            'get_matrix_col_minus_s': ,
            '_get_vec_minus_s': ,
            }

    kwargs = {'get_eta1': ,
            'get_eta2': ,
            'get_poisson_matrix': ,
            'logfactorial': ,
            'logsum': ,
            'f0': ,
            'fn': ,
            'get_matrix_col_minus_s': ,
            '_get_vec_minus_s': ,
            }
"""


def gibbs_sqr_init_params(key: PRNGKey, state: State) -> DeviceArray:
    """Pairs with functional_gibbslib.generic_gibbs(f, theta, phi)"""

    # sample from the base exponential distribution
    params = jax.random.exponential(key)


def gibbs_sqr_update_params(key: PRNGKey, state: State, params) -> DeviceArray:
    """Paris with functional_gibbslib.generic_gibbs(f, theta, phi)"""


def exponential_sqr_node_conditional(
    x: Vector, eta1: float, eta2: float, s: Index, Anode_Exp_natural
) -> float:
    """The nodex conditional distribution of the univariate exponential distribution
    p(x_s | theta, phi, x_minus_s) for the Exponential SQR Graphical model as Inouye 2016"""

    return jnp.exp(
        eta1 * x[s - 1] + eta2 * jnp.sqrt(x[s - 1]) - Anode_Exp_natural(eta1, eta2)
    )


def log_exponential_sqr_node_conditional(
    x: Vector, eta1: float, eta2: float, s: Index, Anode_Exp_natural
) -> float:

    """The log exponential sqr node conditional distribution"""

    return eta1 * x[s - 1] + eta2 * jnp.sqrt(x[s - 1]) - Anode_Exp_natural(eta1, eta2)


def ll_exponential_sqr(
    x: Vector, eta1: float, eta2: float, p: Dimension, Anode_Exp_natural
) -> float:

    """The log likelihood of the Exponential sqr distribution. Note this function is not jittable"""
    ll = 0
    for s in range(1, p + 1):
        ll = ll + log_exponential_sqr_node_conditional(x, eta1, eta2, s, Aexp)
    return ll


def ll_f_sqr_unormalized(x: Vector, eta1, eta2, f_0, p) -> float:

    ll = 0

    def body(i, val):
        # val =
        pass


def gen_exponential(phi_exp: Matrix, p: Dimension, nsamples: Dimension):
    """Generate a vector of points according to the univariate exponential distribution
    numpy function, not jittable

    params:
      npseed :
      eta1 : the natural parameter of the univariate exponential distribution

    return:
      x_exp : a p x nsamples dimensional matrix
              each row in x_exp is an eta1 vector
    """

    # eta1 < 0
    lam = -eta1
    # lam > 0
    beta = 1 / lam  # the scale parameter

    # beta > 0

    return scipy.stats.expon.rvs(beta, size=shape, random_state=npseed)


def natural_sqr_likelihood(
    x: Vector, eta1: float, eta2: float, AexpNatural: Callable
) -> float:
    pass
