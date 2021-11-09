#Author Aji Palar
#The Representation and Scoring of Molecular networks

#An Abstract interface for molecular networks

from abc import ABC, abstractmethod
import math

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

