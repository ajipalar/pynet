#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# What is the overlap in the various datasets that may be used for modeling?
import pandas as pd
import numpy as np
from pathlib import Path
import json

#Read in Stukalov data
data_p = Path("../data")
multi_omnics_p = data_p / "multi-proteomics"
summary_json_p = multi_omnics_p / "summary.json"
apms_interactions = multi_omnics_p / '41586_2021_3493_MOESM6_ESM.xlsx'
proteome_infection = multi_omnics_p / '41586_2021_3493_MOESM9_ESM.xlsx'

#Read in Gordon Data

#Read data contents

with summary_json_p.open() as f:
    summary_json = json.load(f)

def swap_keys(d):
    d2 = {}
    for key in d:
        t = d[key]
        d2[t] = key
    return d2
summary_json = swap_keys(summary_json)

def find_key(query, pydict):
    for key in pydict:
        if query in key:
            return key
    return None
effectome_p = multi_omnics_p / summary_json[find_key("effectome", summary_json)]
#effectome_d = pd.read_excel(table5.as_posix(), sheet_name=1)

#APMS Changes multilevel omnics significant interactions

apms_multi = pd.read_excel(apms_interactions, sheet_name=1)
bait_names = apms_multi.iloc[:, 1].unique()
gene_names = apms_multi.iloc[:, 2].unique()


# In[22]:


# Output DataFrame
columns = ['baits', 'n prey genes']
rowname = ['Gordon', 'Stukalov']
dout = pd.DataFrame(columns=columns, index=rowname)

dout.loc['Stukalov', 'baits'] = len(bait_names)
dout.loc['Stukalov', 'n prey genes'] = len(gene_names)


# In[23]:


dout

