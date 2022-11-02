"""Welcome to the graph
n nodes
m edges (n choose 2)
k a number of edges from 0 to m
u a node index
v a node index lt u

"""
################################################################################
import jax.lax as lax
import jax.numpy as jnp
from collections import namedtuple

Queue = namedtuple("Queue", "val i") # implemented as an array of length l with the first position as i

class Node:
    # Belonging to all nodes / node class
    # Nodes are listed from 1 to nnodes
    nnodes = 0

    def __init__(
        self,
        structure=None,
        degree=None,
        adjacent=None,
        out=[],
        pseq=None,
        polymer=None,
        peptides=None,
    ):
        Node.nnodes += 1
        self.dgr = degree
        self.adj = adjacent
        self.pseq = pseq
        self.plymr = polymer
        self.pep = peptides  # List of of [start, end) in pseq coordinates
        self.strc = structure
        self.out = out


class Model:
    """
    Holds the state of the graph
    """

    pass


class Data:
    """
    Holds all the data used for inference
    """

    pass


class Information:
    """
    Holds the prior information
    """


def compute_score(m: Model, d: Data) -> float:
    pass


def add_nodes(model, n):
    """Add n nodes to an empty imp model"""
    for i in range(0, n):
        model.add_particle(i)
    return model


def adjacent_nodes(u, A, p):
    """
    returns an ordered p-length array
    where the entries equal to p represent no edge
    

    convention -1 is trash

    The returned array has -1 where there 
    """

    m = int(0.5 * p * (p-1))

    nodes = (p) * jnp.ones(p)

    def body(i, val):
        return lax.cond(val.A[u, i]==1, 
                       lambda val: V(val.A, val.nodes.at[val.j].set(i), val.j + 1),
                       lambda val: V(val.A, val.nodes, val.j),
                       val)
    
    V = namedtuple("V", "A nodes j")
    val = V(A, nodes, 0)

    val = lax.fori_loop(0, p, body, val)
    #nodes = nodes.at[u].set(-1)
    nodes = val.nodes
    nodes = nodes.at[jnp.where(nodes==u)].set(p) # remove the self node
    return jnp.sort(nodes)

def enqueue(val, q: Queue):
    return Queue(q.val.at[q.i].set(val), q.i + 1)


def dequeue(q: Queue, q_l):
    t = -1 * jnp.ones(q_l)
    t = t.at[0:q_l-1].set(q.val[1:q_l])
    return q.val[0], Queue(t, q.i - 1)

Stack = namedtuple("Stack", "val i")

def pop(s: Stack, s_l) -> tuple[int, Stack]:
    return s.val[s.i-1], Stack(s.val, s.i-1)

def push(val, s: Stack) -> Stack:
    return Stack(s.val.at[s.i].set(val), s.i + 1)

def dfs(v, A, m, p):
    """
    Starting at node u, perform a depth first search
    Perform a depth-first search over A

    u : the starting node
    A : the adjecency matrix
    m : the maximum number of edges

    After specializing on m dfs is jittable
    #1. let S be a stack
    #2. S.push(v)
    #3. while S is not empty
        #4. v = S.pop()
        #5. if v not labeled as discovered
            #6. label v as discovered

            #7. for all adjacent edges from v to w in G.adjacenct_edges(v) do
                # 8. S.push(v)
    """
    
    stack = -1 * jnp.ones(m) # add a node, 
    idx = 0  # stack index

    # G, v

    #1. let S be a stack
    s = Stack(m * jnp.ones(m), 0)

    discoved = jnp.zeros(p, dtype=bool)


    #2. S.push(v)

    s = push(v, s)

    State = namedtuple("State",
                       "s s_l m discovered")

    def true_fun(val):
        # 6. label v as discovered
        discovered = val.discovered.at[val.v].set(1)



    def body(val):
        #4. v = S.pop()
        v, s = pop(val.s, val.s_l)

        #5. if v not labeled as discovered
            #6. label v as discovered

            #7. for all adjacent edges from v to w in G.adjacenct_edges(v) do
                # 8. S.push(v)

    #3. while S is not empty

    val = lax.while_loop(stack_not_empty, body, init)



def stack_not_empty(s: Stack) -> bool:
    return s.i > 0

def bfs(A):
    """
    Perform a breadth first search over A
    jittable
    """
    ...

def is_potentially_connected(A, p) -> bool:
    """
    A : (p, p) adjacency matrix 
    """
    return jnp.sum(A[jnp.tril_indices(p, k=-1)]) > p-2

def connected_compoents(A):
    """
    Return the number of connected components from the adjecency matrix A
    """



