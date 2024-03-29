import abc
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, List, Tuple

class Network(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def deg(i): ...
    @abstractmethod
    def __getitem__(i, j): ...

class AdjNetwork(Network, metaclass=abc.ABCMeta):
    n: Any
    M: Any
    def __init__(self, n) -> None: ...

def get_graph_ordering(g) -> int: ...
def get_graph_from_order(m, n) -> None: ...
def femax(vmax): ...
def edge_combs(emax): ...
def ne(gid, vmax): ...
def base(nedges, vmax): ...
def mantissa(gid, vmax): ...
def next_vertex(i): ...
def prev_vertex(i): ...
def next_edge(e, vertex_n_max): ...
def prev_edge(e, vmax): ...
def generate_eseq(estart, vertex_n_max) -> Generator[Any, None, None]: ...
def permutations(nitems): ...
def generate_graph(gid, vmax): ...
def ugraph_from_elist(elist: List[Tuple[int]]): ...
def test_next_prev(vmax) -> None: ...

class PoissonSQRGM:
    x: Any
    sqrx: Any
    d: Any
    theta: Any
    phi: Any
    def __init__(self, x, theta: Any | None = ..., phi: Any | None = ...) -> None: ...
    def A(self) -> None: ...
    def B(self) -> None: ...
    def node_term(self) -> None: ...
    def log_likelihood(self) -> None: ...
    def rPhi(d, key): ...

def stable_div(a, b): ...
def normal_pdf(y, mu, sigma): ...
def parabola(x, mu, sigma): ...
def ull_normal(y, mu, sigma): ...
def mode(x, y): ...
