import math
from PlotBioGridStatsLib import Any, Array as Array, DataFrame as DataFrame, ProteinName as ProteinName
from mpl_toolkits.mplot3d import Axes3D as Axes3D
from pathlib import Path as Path
from pyext.src.typedefs import GeneID as GeneID, Matrix as Matrix, Vector as Vector
from typing import Callable

Series: Any

def minmaxlen(col): ...

biogrid: Any
arr: Any
x: Any
spec_counts_df: Any

def test_has_entrez(rowlist: list[list[ProteinName]]): ...

mg: Any
query: Any

def get_ncbi_gene_id(name, mg) -> GeneID: ...
def test_ncbi_gene_id(biogrid) -> None: ...

parser: Any

def get_eta1(theta, phi, x_s, s, i): ...
def get_eta2(theta, phi, x_s, s, i) -> None: ...
def logsum(x): ...
def logfactorial(x_si): ...
def Zexp(eta1, eta2): ...
def f0(xsi, eta1, eta2): ...
def fn(xsi, eta1): ...

c: Any
se: Any
z: Any
je: Any
jlax: Any
y: Any
p: int
n: int
Theta: Any
key: Any
Phi: Any
lam: float
X: Any
Dimension: Any
Index: Any
SufficientStatistic: Any

def vec_minus_s(v: Vector, s: Index): ...
def B(X, s, i): ...
def Anode(eta1: float, eta2: float): ...
def sum_off_1(Phi) -> float: ...
def convex(Theta: Vector, Phi: Matrix, X: Matrix, s: Index, i: Index, p: Dimension, lambda_: float) -> float: ...
def p_tilde(x, i): ...

vec: Any
vec_l: Any
i: int

def get_segment1_bounds(i, vec_l): ...
def get_segment2_bounds(i, vec_l): ...

a: Any

def get_choice(index_to_remove, vec_l): ...

gcp: Any
jgc: Any

def get_matrix_col_minus_s(i, phi, p): ...
def get_poisson_matrix(key, p, lam): ...
def set_diag(phi: Matrix, diag: Vector) -> tuple[Matrix, Vector]: ...
def test_matrix_minus_slice(slicef) -> None: ...
def b(x, p): ...

d: Any
s: int
pf: Any
jpf: Any
phi: Any

def Aexp(eta1, eta2): ...

N: int
j: int
n_replicates: int
j_replicates: int
x_vec: Vector
theta_vec: Vector
Eta: Vector
pn: Callable
p0: Callable
pj: Callable
gamma: float
eta1: float
eta2: float

def exponential_node_conditional_hamiltonian(xs, eta1, eta2): ...
def jcomplex(z): ...
def ftest(x): ...

ftestj: Any

def stirling(x): ...

fac: Any
spec_counts: Any

def fun(x, y): ...

fs: int
fig: Any
ax: Any
Y: Any
zs: Any
Z: Any

def f(x): ...

b: int
deltax: Any

def xk(k, n): ...

xkarr: Any
Mn: Any
Tn: Any
S2n: Any

def integrate(dx): ...

dxs: Any
dMn: Any
dTn: Any
dS2n: Any
erf = math.erf
jerf: Any
y = x

def pyiter_to_slist(pyiter: list) -> str: ...
def post_ncbi(genelist): ...

POSTs: Any

def get_geneid_from_respone(response: POSTs) -> list[GeneID]: ...
def test_post_ncbi(biogrid) -> None: ...

genelist: Any
res: Any
y_true: Any
y_score: Any
fpr: Any
tpr: Any
thresholds: Any
prec: Any
recall: Any
pthresh: Any
