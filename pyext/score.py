#Author Aji Palar
#The Representation and Scoring of Molecular networks

#An Abstract interface for molecular networks

from abc, import ABC, abstractmethod

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
    


