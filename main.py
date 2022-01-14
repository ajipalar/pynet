import IMP
import jax
import numpy
import matplotlib
import matplotlib.pyplot as plt
from packaging import version
from pathlib import Path
import scipy
import sys

pyversion = sys.version.split('|')[0]
assert version.parse(pyversion) >= version.parse('3.9.7')


import pyext.src.graph as graph
import pyext.src.get_network_overlap as get_network_overlap 
import pyext.src.ii as ii
import pyext.src.sampling as sampling
import pyext.src.score as score
import pyext.src.typdefs as typdefs
import pyext.src.utils as utils
import pyext.src.vis as vis

import test.test_graph

from utility import data_setup as ds
from utility.meta import pipe

from pyext.src.repl import cl, ls, pwd

home = Path(".")
pyext = home / "pyext"
data = pyext / "data"
corum = data / "corum"
ecoli = data / "ecoli"
genetic_screen = data / "genetic-screen"
sars_cov_2_lip = data / "sars-cov-2-LiP"
sars_cov_2_ppi = data / "sars-cov-2-ppi"
multi_proteomics = data / "multi-proteomics"

utility = home / "utility"












