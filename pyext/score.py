#Author Aji Palar
#The Representation and Scoring of Molecular networks

#An Abstract interface for molecular networks

from abc import ABC, abstractmethod
import graph_tool as gt

from graph_tool.all import graph_draw
import math
import numpy
import scipy
import inspect
from itertools import combinations
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path, PosixPath

import pandas as pd
import scipy
#User
import pynetio as mio
from myitertools import exhaust, forp
import predicates as pred
from utils import doc, ls, psrc

class Network(ABC):
    
    @abstractmethod
    def deg(i):
        """Return the degree of vertex i"""
        pass
    @abstractmethod
    def __getitem__(i,j):
        """Return the contents of the edge i,j"""
    
class AdjNetwork(Network):
      def __init__(n):
          self.n = n
          self.M = np.ndarray((n,n), dtype=int)
    

def get_graph_ordering(g) -> int:
    """For an unweighted undirected graph g
    return an order index m
    where m is in the range [0, 2^Emax)
    such that g0 = empty graph and g_{2^Emax} = fully connected graph
    n = 4 emax = 6
    
        n!/k!(n-k)!
    ne |graphs| m
    ---+-------------------
     0 |   1  | 0 
     1 |   6  |[1, 7)
     2 |   15 |[7, 16)    
     3 |   20 |[16, 36)   
     4 |   15 |   
     5 |   6  |     emax
     6 |   1  |     1

    n = 3 emax = 3
    
    ne |graphs|
    ---+----------+
     0 |   1  |
     1 |   3  |
     2 |   3  | 
     3 |   1  |
     4 |      |
     5 |      |
     6 |   1  |
    | 0  | gm    |ne | 
    | 1  | ()    | 0 | 
    | 2  | (0, 1)| 1 | 
    | 3  | (0, 2)| 1 | 
    | 4  | (0, 3)| 1 | 
    | 5  | (1, 2)| 1 | 
    | 6  | (1, 3)| 1 | 
    | 7  | (2, 3)| 1 | 
    | 8  |              | len(e) | 
    | 9  |              | len(e) | 
    | 10 |              | len(e) | 
    | 11 |              | len(e) | 
    | 12 |              | len(e) | 
    | 13 |              | len(e) | 
    | 14 |              | len(e) | 
    | 15 |              | len(e) | 
    | 16 |              | len(e) | 
    | 17 |              | len(e) | 
    | 18 |              | len(e) | 
    | 0 | ()
    | 1 | (0, 1)
      2 | (0, 2)
      3 | (0, 3)
      4 | (1, 2)
      5 | (1, 3)
      6 | (2, 3)
      7
      8
      9
      10
      11
      12
      13
      14
      15
      16

    m = len(e) + 


    """
    # e is the edge set of g
    # len(e) is the number of edges in g
    # emax is 0.5n(n-1)
    emax = int(0.5*g.num_vertices()*(g.num_vertices() - 1)) 
    ne = g.num_edges()
    
    # emax choose ne (n, k)
    nk = lambda n, k: math.factorial(n)/(math.factorial(k)*math.factorial(n-k))

    m = 0

    print(emax)

    # The ordering algorithm

    #Get the number of combinations for a given |E|

    #m = mbase + order

    #Compute mbase
    mbase = 0

    #Compute order
    order = 0
    
    return mbase + order

def get_graph_from_order(m, n):
    """For an m in [0, 2^(0.5n(n-1)))
       return the graph g_m
    """

    
"""
gid indexes into the sequence of ordered undirected graphs
mantissa indexes into the sequence of ordered subgraphs
with ne edges
"""

def femax(vmax):
    """
    Returns the maximum number of edges in the undirected graph
    """
    return vmax*(vmax - 1) // 2
def edge_combs(emax):
    """
    Returns an edge combination generator
    The value is the number of combinations
    The index is the number of edges
    
    e.g., index 0 1 2 3
          value 1 3 3 1
    """
    return (int(scipy.special.comb(emax, i)) for i in range(emax+1))
            
def ne(gid, vmax):
    """
    Returns the number of edges for a given gid & vmax
    """
    emax = femax(vmax)
    
    #The index of combs is the edge number
    combs = list(edge_combs(emax))
    assert sum(combs) == 2**emax
    
    solutions = 0
    edges = 0
    while gid >= solutions:
        solutions += combs[edges]
        edges +=1
    return edges - 1

def base(nedges, vmax):
    """
    returns the base for nedges and vmax
    The graph id (gid) = base + mantissa
    """
    emax = femax(vmax)
    combs = list(edge_combs(emax))
    return sum(combs[0:nedges])

def mantissa(gid, vmax):
    """
    returns the mantissa for gid, vmax
    gid = base + mantissa
    """
    b = base(ne(gid, vmax), vmax)
    return gid - b

def next_vertex(i):
    """
    Count the next vertex, undefined over vmax
    """
    return i + 1

def prev_vertex(i):
    """
    Count the previous vertex, undefined for i < 1
    """
    return i - 1

def next_edge(e, vertex_n_max):
    """
    Return the next edge in the sequence
    """
    if e[1] < vertex_n_max - 1: return (e[0], e[1]+1)
    else: return (e[0] + 1, e[0] + 2)
    
def prev_edge(e, vmax):
    """
    Count the previous edge in the sequence
    """
    #if e > first_e:
    # (0, 2) -> (0, 1)
    if e[1] > e[0] + 1: return (e[0], e[1] - 1)
    # (1, 2) -> (0, 3)
    else: return (e[0] - 1, vmax - 1)

def generate_eseq(estart, vertex_n_max):
    yield estart
    while estart != (vertex_n_max - 2, vertex_n_max - 1):
        next_e = next_edge(estart, vertex_n_max)
        estart = next_e
        yield estart   

def permutations(nitems):
    perms = []
    for i in range(nitems):
        for j in range(i+1, nitems):
            perms.append((i, j))
    return perms


def generate_graph(gid, vmax):
    """
    Return an ordered elist from a graph id and n vertices
    """
    nedges = ne(gid, vmax)
    b = base(nedges, vmax)
    m = mantissa(gid, vmax)
    
    if gid == 0:
        return []
    else:
        all_edges = list(generate_eseq((0, 1), vmax))
        combinations_iter = combinations(range(len(all_edges)), nedges)
        
        for i, combo in enumerate(combinations_iter):
            #print(i, combo)
            if i == m:
                eseq = []
                #print(combo)
                for idx in combo:
                    eseq.append(all_edges[idx])
                return eseq
            
def test_next_prev(vmax):
    print("fEdges for vmax = {vmax}")
    print(' e      next   prev   pne    npe   b')
    for e in generate_eseq((0, 1), vmax):
        ne = next_edge(e, vmax)
        pe = prev_edge(e, vmax)
        pne = prev_edge(ne, vmax)
        npe = next_edge(pe, vmax)
        b = e == npe
        
        print(e, ne, pe, pne, npe, b)
