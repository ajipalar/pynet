"""
Author Aji Palar
Input Information
Module for handling data I/O and preprocessing
"""
from . import config
from . import utils

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
from scipy.stats import f_oneway
import sys

# from statsmodels.stats.multitest import fdrcorrection

# Local imports
src = pathlib.Path("~/Projects/pynet")

if config.PRINT_MODULE_INFO:
    utils.moduleinfo(locals())


def read_lip_data():
    "Read the LiP data into a pandas data frame"
    dpath_lip = (
        src
        / "data/sars-cov-2-LiP/20210430_084806_lgillet_SARS2_NSP9_LiP_SpecLib_Report.xls"
    )
    # dpath_TC = "../data/sars-cov-2-LiP/20210430_084629_lgillet_SARS2_NSP9_TC_SpecLib_Report.xls"
    return pd.read_csv(dpath_lip, delimiter="\t")


def format_lip_data(df):
    df.rename({"PG.ProteinAccessions": "uids"})
    return df


def summarize_lip_data():
    return nproteins, npeptides


def summarize_gordon_data():
    return nproteins


def read_stuk_apms_data():
    dpath_stuk = pathlib.Path("")


def read_gordon_data():
    pass


def summarize_stuk_data():
    return nAPMSproteins, nRNAgenes, nUbiquit, nPhos


def overlap(a, b):
    """Returns the magnitude of the intersection between two sets."""
    return len(a.intersection(b))


def get_dataset_overlap(dataset_dict, dataset_names=None):
    """For n datasets in param: dataset_dict, returns a
    symmetric intersection matrix where the values of the
    matrix are the pairwise intersection between two datasets
    in the dataset_dict.

    returns an intersection_matrix: np.ndarray(dtype=np.uint)"""

    if not dataset_names:
        dataset_names = [
            "LiP",
            "APMS_Gordon",
            "APMS_Stuk",
            "Total_Stuk",
            "mRNA_Stuk",
            "Phos_Stuk",
            "Ubiq_Stuk",
        ]
    ndatasets = len(dataset_names)
    intersection_matrix = np.ndarray((ndatasets, ndatasets), dtype=np.uint)
    for i, key in enumerate(dataset_names):
        for j, key2 in enumerate(dataset_names):
            intersection_matrix[i, j] = overlap(dataset_dict[key], dataset_dict[key2])
    return intersection_matrix
