import inspect
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd



#User
import tools.io as myio
from tools.myitertools import exhaust, forp
import tools.predicates as pred
from tools.utils import doc, ls, psrc


home = Path.home()
src = home / 'Projects/pynet'
data = src / 'data'
lip = data / 'sars-cov-2-LiP'
multi = data / 'multi-proteomics'

#Variables for testing lip funcitons

xls_paths = list(myio.gen_excelpaths_from_dir(lip))
lip1 = xls_paths[1]

lipgen = myio.parse_lip_xsl_file(lip1)

