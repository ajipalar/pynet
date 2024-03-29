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


def adjacent_nodes(u, A, p) -> tuple:
    """
    Find the adjacent nodes of an edge

    Args:
      u : int a node in the graph
      A : the p x p adjacency matrix
      p : int 

    Returns:
      arr : a p length array whose entries are the adjacent vertices
      degree : the degree of the node
    """
    Variants = namedtuple("Variants", "nodes degree j")
    Invariants = namedtuple("Invariants", "A p m u")
    State = namedtuple("State", "iv vi")
    Val = namedtuple("Val", "state i")

    m = p * (p - 1) // 2 
 
    iv = Invariants(A=A,
                    p=p,
                    m=m,
                    u=u)

    nodes = p * jnp.ones(p, dtype=int)

    vi = Variants(nodes=nodes,
                  degree=0,
                  j=0)

    state = State(iv=iv, 
                  vi=vi)

    #adjacent_nodes = jax.tree_util.Partial(adjacent_nodes)

    def cond_fun(val):
        p1 = val.state.iv.A[val.state.iv.u, val.i] == 1
        p2 = val.state.iv.u != val.i
        return p1 & p2

    def true_fun(val):
        nodes = val.state.vi.nodes.at[val.state.vi.j].set(val.i)
        j = val.state.vi.j + 1
        degree = val.state.vi.degree + 1
        vi = Variants(nodes, degree, j)
        state = State(iv, vi)
        
        return Val(state, val.i)


    def body(i, state):
        val = Val(state, i)
        val = lax.cond(cond_fun(val), 
                       true_fun, 
                       lambda x: x,
                       val)
        return val.state
    
    state = lax.fori_loop(0, p, body, state)
    #nodes = nodes.at[u].set(-1)
    return jnp.sort(state.vi.nodes), state.vi.degree

def enqueue(val, q: Queue):
    return Queue(q.val.at[q.i].set(val), q.i + 1)


def dequeue(q: Queue, q_l):
    t = -1 * jnp.ones(q_l)
    t = t.at[0:q_l-1].set(q.val[1:q_l])
    return q.val[0], Queue(t, q.i - 1)

Stack = namedtuple("Stack", "val i")

def pop(s: Stack) -> tuple[int, Stack]:
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
    

    # G, v

    #1. let S be a stack
     # A node can only be discovered once
     # There are p nodes
     # During any discovery up to p-1 additional nodes may be added to the stack
     # Before adding nodes onto the stack a node is removed
     # therefore the max stack size is p * (p-1) - p
     # duplicate vertices may be placed on the stack
     # let n be the number of vertices and m be the maximum number of edges given n
     # what is the maximum number of duplicate vertices?
     # Fully connected graph, each vertex has n-1 adjacent vertices
     # So 

    # Allocate a stack

    Invariants = namedtuple("Invariants",
                            "p A m")
    Variants = namedtuple("Variants",
                          "s discovered v")

    State = namedtuple("State",
                       "iv vi")
    s_l = p * (p - 1) 

    s = p * jnp.ones(s_l, dtype=int)
    discovered = jnp.zeros(p, dtype=bool)
    
    s = Stack(s, 0)
    s = push(v, s)

    vi = Variants(s=s,
                  discovered=discovered,  # none 
                  v=v)

    iv =  Invariants(p=p,
                     A=A,
                     m=m)

    state = State(iv=iv, 
                  vi=vi)

    def s_not_empty(state) -> bool:
        return state.vi.s.i > 0


    def dfs_loop(state):
        #4. v = S.pop()
        v, s = pop(state.vi.s)
        vi = Variants(s=s, 
                      discovered=state.vi.discovered,
                      v=v)
        state = State(iv=iv,
                      vi=vi)
        return lax.cond(state.vi.discovered[state.vi.v]==False, true_fun, lambda x: x, state)
        

    def true_fun(state):
        W = namedtuple("WV", "degidx adj deg")


        def while_condf(v):
            state, w = v
            return w.degidx < w.deg

        def body_fun(v):
            state, w = v
            
            u = w.adj[w.degidx]
            s = push(u, state.vi.s)

            state = State(iv=state.iv,
                          vi=Variants(s, state.vi.discovered, state.vi.v))

            w = W(degidx=w.degidx + 1,
                  adj=w.adj,
                  deg=w.deg)

            return state, w


        # 6. label v as discovered
        discovered = state.vi.discovered.at[state.vi.v].set(True)
        state = State(iv=state.iv,
                      vi=Variants(state.vi.s,
                                  discovered,
                                  state.vi.v))

        adj_nodes, degree = adjacent_nodes(state.vi.v, state.iv.A, p)

        w = W(degidx=0,
              adj=adj_nodes,
              deg=degree)
        state, _ = lax.while_loop(while_condf, body_fun, (state, w))
        return state

    state = lax.while_loop(s_not_empty, dfs_loop, state)
    return state




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



