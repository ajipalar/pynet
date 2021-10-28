import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd



#User
import tools.excelio as eio
from tools.utils import doc, ls
from tools.myitertools import exhaust, forp

home = Path.home()
src = home / 'Projects/pynet'
data = src / 'data'
lip = data / 'sars-cov-2-LiP'
multi = data / 'multi-proteomics'
