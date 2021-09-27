"""
Welcome to the graph
"""

class Node():
    # Belonging to all nodes / node class
    nnodes = 0
    def __init__(self, structure=None, degree=None, adjacent=None, pseq=None, polymer=None, peptides=None):
        Node.nnodes += 1
        self.dgr = degree
        self.adj = adjacent
        self.pseq = pseq
        self.plymr = polymer
        self.pep = peptides #List of of [start, end) in pseq coordinates
        self.strc = structure

class Model():
    pass
