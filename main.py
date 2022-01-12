import sys

pyversion = sys.version
assert pyversion[0:5] == '3.9.7'


from pathlib import Path
from utility import data_setup as ds
from utility.data_setup import ls
from pyext.repl import cl, pwd
import test.test_graph

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



