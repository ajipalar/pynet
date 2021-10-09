import numpy as np
import jax.numpy as jnp

adj_mtrx = np.zeros((10,10))
adj_mtrx[0, 1] = 1
adj_mtrx[1, 2] = 1
adj_mtrx[2, 5] = 1
adj_mtrx[5, 9] = 1
adj_mtrx[9, 3] = 1
adj_mtrx[3, 7] = 1
adj_mtrx[2, 3] = 1

def score(i,j, am):
    return -np.log(distance(i,j, am))

def priority_insert(obj, key, queue):
    """
    insert object and key to queue
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

def adj(s, am):
    """
    s: source vertex
    am: adjacency matrix
    return: adjacent vertices
    """
    return np.where


def dijkstra(s, t, adj):
    l = len(adj)
    path_weight = np.array(np.inf*l)
    path_weight[s] = 0
    previous = np.zeros(l) 
    #Priority queue with two arrays and ascociated functions
    #    node 0, node 1, ... node l-1
    #key
    #obj
    remaining = np.zeros((2,l)) 
    
    
def distance(s,t, am):
    """
    s: source vertex
    t: target vertex
    am: adjacency matrix
    return: d (distance)
    """
    return d
    

