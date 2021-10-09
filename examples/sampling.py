import numpy as np
import jax.numpy as jnp

A = np.zeros((10,10), dtype=np.int32)
A[0, 1] = 1
A[1, 2] = 1
A[2, 5] = 1
A[5, 9] = 1
A[9, 3] = 1
A[3, 7] = 1
A[2, 3] = 1

#Ground Truth
G = np.ones((10, 10))

M = np.zeros((10, 10))
def score(i,j, A):
    return -np.log(distance(i,j, A))

def harmonic_restraint(d, mu, s=1.0):
    """
    Penalize distances away from mu
    """
    return (np.abs(d - mu)/s)**2
    
def priority_insert(obj, key, queue):
    """
    insert object and key to priority queue
    """
    pass

def priority_delete(key, queue):
    """
    Delete key:obj from priority queue
    """
    pass

def priority_remove_min(queue):
    """
    remove object with minimum key
    """
    pass

def priority_decrement(key, queue):
    """
    decrease the key
    """
    pass

def adj(s, A):
    """
    s: source vertex
    A: Adjacency matrix
    return: adjacent vertices
    """
    return np.where


def dijkstra(s, t, A):
    l = len(A)
    path_weight = np.array(np.inf*l)
    path_weight[s] = 0
    previous = np.zeros(l) 
    #Priority queue with two arrays and ascociated functions
    #    node 0, node 1, ... node l-1
    #key
    #obj
    remaining = np.zeros((2,l)) #A priority queue
    while len(remaining) > 0:
        pass
    return distances #An array of distances from s to every other node where distances[t] = d

    
    
def distance(s,t, A):
    """
    s: source vertex
    t: target vertex
    A: Adjacency matrix
    return: d (distance)
    """
    return d

def dgrs(A):
    """
    Gives the degree of every vertex in A
    """
    return A.sum(axis=1)

def dgr(s, A):
    """
    Gives the degree of node i
    """
    return A[s].sum()
