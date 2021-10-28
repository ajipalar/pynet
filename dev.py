import inspect
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd



#User
import tools.io as io
from tools.myitertools import exhaust, forp
import tools.predicates as pred
from tools.utils import doc, ls


home = Path.home()
src = home / 'Projects/pynet'
data = src / 'data'
lip = data / 'sars-cov-2-LiP'
multi = data / 'multi-proteomics'
