import numpy as np
import jax.numpy as jnp

adj = np.zeros((10,10))
adj[0, 1] = 1
adj[1, 2] = 1
adj[2, 5] = 1
adj[5, 9] = 1
adj[9, 3] = 1
adj[3, 7] = 1
adj[2, 3] = 1

def score(i,j):
    return -np.log(distance(i,j))

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
    
    
def distance(i,j, adj):
    pass

