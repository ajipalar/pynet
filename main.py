"""
# During development run >>> from main import *
in the python interpreter from the pynet home directory
"""
import pyext.src.ii as ii

import pyext.src.mcmc as mcmc
import pyext.src.score as score
import pyext.src.typedefs as typedefs
import pyext.src.utils as utils
import pyext.src.vis as vis

import test.test_graph

from pyext.src.myitertools import forp, exhaust
from utility import data_setup as ds
from utility.meta import pipe

from pyext.src.repl import cl, ls, pwd
from pyext.src.project_paths import (
    home,
    pyext, 
    data,
    corum,        
    ecoli,
    genetic_screen,
    sars_cov_2_lip,
    sars_cov_2_ppi,
    multi_proteomics,
    synthetic, 
    utility 
)
import pyext.src.graph as graph
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


#import pyext.src.get_network_overlap as get_network_overlap
