#Author Aji Palar
#Data I/O, Data formatting and preprocessing
#Creation and handling of input information data structres used during modeling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from statsmodels.stats.multitest import fdrcorrection

def read_lip_data():
    dpath_lip = "../data/sars-cov-2-LiP/20210430_084806_lgillet_SARS2_NSP9_LiP_SpecLib_Report.xls"
    dpath_TC = "../data/sars-cov-2-LiP/20210430_084629_lgillet_SARS2_NSP9_TC_SpecLib_Report.xls"
    return pd.read_csv(dpath_lip, delimiter="\t")

def summarize_lip_data():
    return nproteins, npeptides

def summarize_gordon_data():
    return nproteins

def summarize_stuk_data():
    return nAPMSproteins, nRNAgenes, nUbiquit, nPhos

def overlap(a, b):
    return len(a.intersection(b))
def get_dataset_overlap(dataset_dict, dataset_names=None):
    """
    :param dataset_dict: datasets in set format
    :return:
    """
    if not dataset_names:
        dataset_names = ['LiP', 'APMS_Gordon', 'APMS_Stuk', 'Total_Stuk',
                         'mRNA_Stuk', 'Phos_Stuk', 'Ubiq_Stuk']
    ndatasets = len(dataset_names)
    overlap_matrix = np.ndarray((ndatasets, ndatasets), dtype=np.uint)
    for i, key in enumerate(dataset_names):
        for j, key2 in enumerate(dataset_names):
            overlap_matrix[i, j] = overlap(dataset_dict[key], dataset_dict[key2])
    return overlap_matrix