import numpy as np
import jax.numpy as jnp
from typing import NewType

Dist = NewType('Dist', float)
EdgeID = NewType('EID', int)
VertexID = NewType('VID', int)

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
def rsum(v):
    r=0
    while v > 0:
        v-=1
        r+=v
    return r
def harmonic_restraint(d, mu, s=1.0):
    """
    Penalize distances away from mu
    """
    return (np.abs(d - mu)/s)**2
def degree_prior(degree, prior_degree, s=1.0):
    """
    Harmonic well over degrees centered at prior
    """
    return ((np.abs(degree - prior_degree)/s)**2).sum()
    
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
    s: src vertex
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
    
def distance(s,t, A)-> Dist:
    """
    s: src vertex
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

def vmax(A):
    return A.shape[0]
def e_base(s, v):
    r=0
    vmax = v
    while v > vmax -s:
        v-=1
        r+=v
    return r
def revert_sum(n):
    steps = 0
    i=1
    while i < n:
        pass
    return steps
def get_immutable_edge_id(s, t, vmax):
    """
    Note s < t
    s: src
    t: target
    vmax: max vertices
    return edge_id
    """
    return e_base(s, vmax) + t - s - 1

def get_edge_from_id(eid, vmax):
    #eid [0 -> 1/2v(v-1) -1]
    s = 0 #[0 -> vmax -2]
    t = 1 #[1 -> vmax -1]
    c = 0
    while eid > vmax:
        #Reduce the eid by c
        eid = eid - s*c
        c+=1
    return s, eid

def src_from_eid(eid: EdgeID, v) -> VertexID:
    """
    How many times does v go into eid?
    """
    u = v-2
    count = 0
    while eid > u:
        count+=1
        u = u + v - 1 - count
    return count

def smart_mod(a, b):
    if b != 0:
        return a % b
    else:
        return 0
def count(eid, v):
    eid = base + leftover
def get_immutable_edge_id(s, t, v):
    return s*v+t
    pass
def get_immutable_vertices(eid, A):
    pass
def transition_kernal(A, n=4):
    """
    Define the moves for A
    The Proposal distribution
    """
    shape = A.shape
    #Randomly change 1/n of the edges
    v = vmax(A)
    Emax = v*(v - 1)//2
def integral_lower_tri(x:int, v:int)-> int:
    """
    x: The base 1 index of the triangle
    note x = vertex id + 1
    Call integral_lower_tri(vid+1, v)
    """
    return int((-x**2)/2 + x*v - 0.5*x)
def edge_from_eid(eid, v):
    """
    v: the number of vertices in the graph
    eid: the immutable edge id
    return s, t : src target
    """
    #Area of the lower triangle not including self loops
    s = src_from_eid(eid, v)
    area = integral_lower_tri(s, v)
    eid = eid - area + s + 1
    return s, eid    
    
def MH_MCMC(A, steps=10000, score_f=None, SA=1, Q=None):
    #The target distribtuion
    target = score_f**SA
    #The transition kernel
    Q = Q
    
    pass
