"""
Welcome to the graph
"""

class Node():
    # Belonging to all nodes / node class
    # Nodes are listed from 1 to nnodes
    nnodes = 0
    def __init__(self, structure=None, degree=None, adjacent=None, out = [], pseq=None, polymer=None, peptides=None):
        Node.nnodes += 1
        self.dgr = degree
        self.adj = adjacent
        self.pseq = pseq
        self.plymr = polymer
        self.pep = peptides #List of of [start, end) in pseq coordinates
        self.strc = structure
        self.out = out

class Model():
    """
    Holds the state of the graph
    """
    pass

class Data():
    """
    Holds all the data used for inference
    """
    pass
class Information():
    """
    Holds the prior information
    """
def compute_score(m: Model, d: Data) -> float:
    pass