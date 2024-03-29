"""
A numpy compatible model prototype not focused on performance

Traceable functions require shape information
statically known at compile time

Static arguments must be traced from the top level
functions through all sub functions.

A function parameter is annotated as static
with the static_param_name naming convention

"""

import numpy as np
import scipy as sp
from scipy.special import comb
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from collections import deque, namedtuple
from itertools import combinations, chain
import matplotlib.pyplot as plt
from functools import partial
from jax.tree_util import Partial

import lpdf
import pynet_rng

# mini jax `objects`
AdjacentNodes = namedtuple("AdjacentNodes", "arr n_adjacent_nodes")
Queue = namedtuple("Queue", "arr tail_idx length")
JList = namedtuple("JList", "arr lead_idx")

def create_empty_jlist(length, dtype):
    """
    Creates an empty JList
    The array is implements with large default values
    """
    return JList(jnp.ones(length, dtype=dtype) * length + 2, lead_idx=0)

def append_jlist(jlist, val):
    arr = jlist.arr.at[jlist.lead_idx].set(val) 
    return JList(arr, jlist.lead_idx + 1)

def pop_jlist(jlist):
    """
    pop an element off the end of the list
    Returns:
      val : the popped element
      jlist : a jlist the same size as the original
    """
    val = jlist.arr[jlist.lead_idx-1]
    return val, JList(jlist.arr, jlist.lead_idx - 1)

def in_jlist(x, jlist) -> bool:
    """
    Tests if the value x is in jlist
    """

    lead_idx = jlist.lead_idx
    in_ = False
    def body(i, val):
        in_ , jlist , x= val
        in_ = (x == jlist.arr[i]) | in_
        return in_, jlist, x

    val = in_, jlist, x
    val = jax.lax.fori_loop(0, lead_idx, body, val) 
    in_, jlist, x = val
    return in_


def _assert_A(A):
    """
    Checks the assumptions of A
    """
    n = len(A)
    assert A.shape == (n, n), f"A is not square. shape: {A.shape}"
    assert A.dtype in [np.int8, np.int32, np.int64], f"A is of dtype {A.dtype}"
    assert np.all(np.diag(A) == 0), f"A has non zero diagonal elements"
    assert np.sum(A > 1) == 0, f"A has entries greater than 1"
    assert np.sum(A < 0) == 0, f"A has entreis less than 0"

def _assert_Cs(Cs):
    n = len(Cs)
    assert Cs.shape == (n,)
    assert Cs.dtype in [np.int8, np.int32, np.int64], f"A is of dtype {A.dtype}"
    assert n > 0
    assert np.sum(Cs < 0) == 0
    assert Cs[0] == 0, f"A composite must contain the bait as the first element"
    assert np.all(np.array(sorted(Cs)) == Cs)

def get_Asc_from_A(As, Cs):
    """
    Get the Asc matrix from As and Cs
    Params:
      As - the sth adjacency matrix of possible prey and the bait
      Cs - the composite vector in the As coordinates.
    Returns:
      Asc - the sth adjacency matrix at the bait and prey of Cs
    """

    _assert_Cs(Cs)
    _assert_A(As)
    Asc = As[:, Cs]
    Asc = Asc[Cs, :]
    _assert_A(Asc)

    return Asc

def d(As, Cs)-> int:
    """
    The disconectivity function counts the number of prey disconnected to the bait
    The function assumes (without checking) that the bait is node 0.

    Parameters:
      As   : the sth adjacency matrix of possible prey and the bait
      Cs  : composite vector in As coordinates
    Returns:
      n_disconnected_prey : the number of disconnected prey  
    """

    _assert_Cs(Cs)
    _assert_A(As)
    Asc = get_Asc_from_A(As, Cs)
    n = len(Asc)
    n_prey = n-1
    connected_composite: set = bfs(Asc, 0)
    n_connected_prey = len(connected_composite) - 1 # subtract one because bait is included
    n_disconnected_prey = n_prey - n_connected_prey
    assert 0 <= n_disconnected_prey < n, f"{n_disconnected_prey}"
    return n_disconnected_prey


def get_adjacent_nodes(A, node):
    """
    Get all nodes adjacent to 'node' from an adjacency matrix 'A'
    Params:
      A an adjacency matrix
    Returns:
    """

    n = len(A)
    assert node < n
    assert type(node) in [np.int8, np.int32, np.int64, int] 
    _assert_A(A)

    column = A[:, node] 
    assert column.shape == (n,), f"columns shape: {column.shape}"


    # numpy selecting works, maybe not in jax?

    # This works because the self node must equal 0

    adjacent_nodes = np.where(column == 1)[0]

    return adjacent_nodes

def get_adjacent_nodes_jax(A, node_index: int, len_A, node_dtype=jnp.int32):
    """
    Traceable version of `get_adjacent_nodes`
    
    Params:
      A - an adjacency matrix
      node_index - the node from which adjacent nodes are to be found
      len_A - the number of rows/columns in A
    Returns:
      AdjacentNodes: namedtuple  
        arr: a sorted_array of adjacent_nodes
        n_adjacent_nodes: the number of adjacent nodes
    """

    # The max size is len_A-1
    # The garbage values are placed at the end of the array
    # The garbage values are  len_A * 9999 + 1

    default_values = len_A + 1 

    adjacent_nodes = jnp.ones(len_A-1, dtype=node_dtype) * default_values

    left_to_diag_len = node_index
    diag_to_bot_len = len_A - node_index - 1
    # Get the adjacent nodes by row
    
    def by_row(i, val):
        adjacent_nodes, A = val
        query_val = A[node_index, i]
        adj_val = jax.lax.cond(query_val == 1,
                               lambda i: i,
                               lambda i: default_values,
                               i)

        adjacent_nodes = adjacent_nodes.at[i].set(adj_val)
        return adjacent_nodes, A

    adjacent_nodes, A = jax.lax.fori_loop(0, left_to_diag_len, by_row, (adjacent_nodes, A))

    def by_col(i, val):
        adjacent_nodes, A = val
        query_val = A[i, node_index]
        adj_val = jax.lax.cond(query_val == 1,
                               lambda i: i,
                               lambda i: default_values,
                               i)
        adjacent_nodes = adjacent_nodes.at[left_to_diag_len + i].set(adj_val)
    
        return adjacent_nodes, A

    # Get the adjacent nodes by column
    adjacent_nodes, A = jax.lax.fori_loop(node_index+1, len_A, by_col, (adjacent_nodes, A))
    adjacent_nodes = jnp.sort(adjacent_nodes)

    n_adjacent_nodes = jnp.sum(adjacent_nodes != default_values) 
    return AdjacentNodes(adjacent_nodes, n_adjacent_nodes)

def bfs(A, root: int):
    """
    A breadth first search

    Params:
      A an adjacency matrix
    Returns:
      explored: the set of nodes connected to the root
    """
    _assert_A(A)
    assert type(root) in [np.int8, np.int32, np.int64, int] 

    n = len(A)

    explored = set()
    explored.add(root)

    m = n * (n - 1) // 2

    # a queue of nodes
    q = deque([root], maxlen=n)

    while len(q) > 0:
        v = q.popleft()
        adjacent_nodes = get_adjacent_nodes(A, v)
        for node in adjacent_nodes:
            if node not in explored:
                explored.add(node)
                q.append(node)
    return explored



def create_empty_queue(length, dtype):
    # Queue
    # The queue has a size equal to the number of elements in the queue
    # The queue has a length equal to len(arr)
    arr = jnp.zeros(length, dtype=dtype)
    # The empty queue's leader is at position 0 
    leader_idx = 0
    # The empty queue's tail is at position 0

    return Queue(arr=arr, tail_idx=0, length=length)

def enqueue_jax(q, val):
    """
    A queue is a first in first out data structure
    This function is meant to traced by JAX and jit compiled 
    If the queue is full (q.length) and enqueue_jax is called,
    enqeue_jax is undefined and the queue is invalid
    """
    return Queue(q.arr.at[q.tail_idx].set(val), q.tail_idx + 1, q.length)

def dequeue_jax(q):
    """
    dequeue a Queue. traceable 
    """
    arr = jax.lax.fori_loop(1, q.length, lambda i, arr: arr.at[i-1].set(arr[i]), q.arr) 
    return q.arr[0], Queue(arr, q.tail_idx-1, q.length)




BFS_WHILE_PARAMS = namedtuple("BFS_WHILE_PARAMS",
                              "explored q A len_A")

def _bfs_jax_while_loop_piece(explored, q, A, len_A):
    body_fun = Partial(_bfs_jax_while_body, len_A=len_A)
    return jax.lax.while_loop(
            lambda x: x[1].tail_idx > 0,
            lambda x: body_fun(*x),
            (explored, q, A))


def _bfs_jax_while_body(explored, q, A, len_A):
#    jax.debug.breakpoint()
    v, q = dequeue_jax(q)

    adj = get_adjacent_nodes_jax(A, v, len_A)
    _, explored, q, adj = _bfs_jax_fori_loop_piece(
            adj, explored, q)
#    jax.debug.breakpoint()

    return explored, q, A 



def _bfs_jax_fori_loop_body(i, val):
    """ (node, (explored, q)) -> (explored, q) """
    _, explored, q, adj = val
    node = adj.arr[i]
    init = node, explored, q
    #jax.debug.breakpoint()

    in_: bool = in_jlist(node, explored) 
    init = jax.lax.cond(
        in_, 
        lambda x: init, # do nothing
        lambda x: _bfs_jax_true_fun(*init), # update explored  q
        init)
    _, explored, q = init
    #jax.debug.breakpoint()
    return node, explored, q, adj

def _bfs_jax_fori_loop_piece(adj: AdjacentNodes, explored: JList, q: Queue):
    """ (adj, explored, q) -> (explored, q)"""
    val = 0, explored, q, adj
    return jax.lax.fori_loop(0, adj.n_adjacent_nodes,
                             _bfs_jax_fori_loop_body,
                             val)

def _bfs_jax_true_fun(node: int, explored: JList, q: Queue):
    """ (T) -> (T) """
    explored = append_jlist(explored, node)
    q = enqueue_jax(q, node)
    return node, explored, q


    
def bfs_jax(A, root: int, len_A, node_dtype=jnp.int32):
    """
    A jittable breadth first search

    Params:
      A: an adjacency matrix
    Returns:
      explored: the set of nodes connected to the root
    """
    
    # An array of explored nodes
    explored = create_empty_jlist(len_A, dtype=node_dtype) 

    # Assign the bait (root) as found
    explored = append_jlist(explored, root)


    # While the queue is not empty
#    jax.lax.while_loop(lambda q: q.tail_index != 0, while_loop_body, 

    q = create_empty_queue(length=len_A, dtype=node_dtype)
    q = enqueue_jax(q, root)

    explored, q, A = _bfs_jax_while_loop_piece(explored, q, A, len_A)
#    jax.debug.breakpoint()
    return explored
    







def C(Ss, ts, node_dtype=jnp.int32):
    """
    The composite construction function
    Given an array of saint scores Ss in the As coordinate system and
    a threshold ts, construct a composite whose members have a score above ts
    """
    assert 0 <= ts <= 1
    n_possible_prey = len(Ss)
    assert n_possible_prey > 1

    prey = np.where(Ss >= ts)[0]
    prey = prey + 1  # index to As coordinates
    n_prey = len(prey)
    Cs = np.zeros(n_prey + 1, dtype=node_dtype)
    Cs[1:] = prey
    return Cs

def enumerate_prey_composites(n_possible_prey) -> iter:
    """
    Returns an iterable of composites in the As coordinates.
    Each composite is missing the bait
    """
    assert n_possible_prey > 1
    it = iter(())
    for n_prey in range(1, n_possible_prey + 1):
        it = chain(it,
             combinations(range(1, n_possible_prey + 1), n_prey))  
    return it

def magnitude_omega(n_possible_prey) -> int:
    """
    Returns the size of the sample space of possible composites 
    """
    return int(2**n_possible_prey -1)

def enumerate_Ss(Ss):
    """
    This O(n) algorithm where n = len(Ss) counts the number of composites
    among all possible composites for a given set of possible prey with a
    minimal SAINT score of X

    Params:
      Ss - the saint score prey array where the first element of the array correspond to
      a prey id of 1.

    Returns:
      event_frequency - an m length array of event frequencies corresponding to score_set
      score_set - an m length array of saint outcome scores corresponding to event_frequency
    """
    n_possible_prey = len(Ss)

    assert Ss.shape == (n_possible_prey,)
    assert n_possible_prey > 1

    score_set = sorted(set(Ss))


    # The total number of events is the nth row of pascals triangle
    # equal to 2**n minus the first (empty) event
    total_events = int(2**n_possible_prey - 1) 

    n_outcomes = len(score_set)

    outcome_frequency = np.zeros(n_outcomes)

    for i, min_score in enumerate(score_set):
        assert 0 < min_score <= 1

        n_prey_gt_min_score = len(np.where(Ss > min_score)[0])
        n_prey_gte_min_score = len(np.where(Ss >= min_score)[0])

        n_events_gt_min_score = int(2**(n_prey_gt_min_score) - 1)
        n_events_gte_min_score = int(2**(n_prey_gte_min_score) - 1)

        assert n_events_gt_min_score < n_events_gte_min_score, f"gt: {n_events_gt_min_score, n_events_gte_min_score}"

        n_events_eq_min_score = n_events_gte_min_score - n_events_gt_min_score
        outcome_frequency[i] = n_events_eq_min_score

    return outcome_frequency, np.array(score_set)

def log_pdf_Cs__Ss_ts_builder(Ss) -> float:
    """
    Build log p(Cs|Ss)  aka log p(Cs|Ss, ts)

    This function builds the function log_pdf_Cs__Ss_ts aka
    the composite identity restraint.

    Params:
      Ss - A SAINT score array
    Returns:
      log_pdf_Cs__Ss_ts - a log probability mass funciton over composites
    """

    n_possible_prey = len(Ss)
    outcome_frequency, score_set = enumerate_Ss(Ss)
    n_outcomes = magnitude_omega(n_possible_prey)
    assert sum(outcome_frequency) == n_outcomes

    outcome_probabilities = outcome_frequency / n_outcomes
    for prob in outcome_probabilities:
        assert 0 < prob <= 1 

    log_probabilities = np.log(outcome_probabilities)

    # create the function
    outcome_map = {score_set[i]: log_probabilities[i] for i in range(len(score_set))}

    def log_pdf_Cs__Ss_ts(Cs, Ss) -> float:
        """
        The function fs taks as input a composite and a saint score array
        and outputs the probability of the composite
        """

        min_saint_score = 1
        n_possible_prey = len(Ss)
        n_composite_members = len(Cs)
        assert Cs.shape == (n_composite_members,)
        assert Cs[0] == 0, f"{Cs[0]}"
        assert len(Cs) > 1, f"{len(Cs)}"
        
        for cs_index in range(1, len(Cs)):
            node = Cs[cs_index]
            assert 0 < node <= n_possible_prey, f"{node, n_possible_prey}"

            ss_index = node - 1
            saint_score = Ss[ss_index]
            min_saint_score = min(min_saint_score, saint_score)

        log_prob = outcome_map[min_saint_score]
        return log_prob

    return log_pdf_Cs__Ss_ts


def log_pdf_As__Cs_lambda_s(As, Cs, lambda_s: float):
    """
    The composite connectivity restraint. A log prior conditional density
    Params:
      As  the sth adjacency matrix
      Cs  the sth composite
      lambda_s  the sth exponential rate parameter
    returns:
      score  the restraint score. a float
    """
    a = d(As, Cs)
    return np.log(lambda_s)  -  a*lambda_s


def log_pdf__M_D_I_restraint_builder(Ss, log_pdf_ts):
    """
    The prior density p(M|D, I) is a product of terms exp^g(s) 
    log p(M|D, I) = Sum g_s(A_s, C_s, S_s, t_s, mu_s, lambda_s)
    across the sth spanning composite. This function builds the sth term

    """

    log_pdf_Cs__Ss_ts = log_pdf_Cs__Ss_ts_builder(Ss)

    def g_s(As, Cs, ts, mu_s, lambda_s):

        log_prob_ts = log_pdf_ts(ts)
        log_prob_Cs__Ss_ts = log_pdf_Cs__Ss_ts(Cs, Ss)
        log_prob_As__Cs_lambda_s = log_pdf_As__Cs_lambda_s(As, Cs, lambda_s)

        return np.array([log_prob_As__Cs_lambda_s,
                         log_prob_Cs__Ss_ts,
                         log_prob_ts])  # missing p(mus)

    docstring = """The sth log prior restraint"""

    g_s.__doc__ = docstring
    return g_s

def r(Sigma_inv_s, As) -> float:
    """
    The root mean square deviation funciton between As and Sigma_inv_s
    evaluated where off diagonal As == 0
    """

    As = As[1:, 1:]
    n = len(As)
    assert n > 1
    As = np.tril(As, k=-1)
    U = np.triu(np.ones((n, n)))
    As = As + U
    assert sum(np.diag(As)) == n
    i_s, j_s = np.where(As == 0)

    mag_z = len(i_s)

    assert np.all(i_s > j_s)
    
    r = np.sqrt(np.sum(Sigma_inv_s[i_s, j_s]) / mag_z)

    return r

def log_pdf_yrs__mus_Sigma_inv_s(yrs, mu_s, Sigma_inv_s):
    """
        
    """
    return lpdf.multivariate_normal(yrs, mu_s, Sigma_inv_s)

def log_pdf_Sigma_inv_s__As_alpha_s(Sigma_inv_s, As, alpha_s) -> float:
    """
    The exponential coupling distribution distance restraint
    """

    return np.log(alpha_s) - r(Sigma_inv_s, As) * alpha_s

def move_As(key, As):
    ...

def move_Sigma_inv_s(key, Sigma_inv_s):
    ...

def move_ts(key, ts):
    ...


def get_possible_edges(len_A):
    possible_edges = jnp.array(list(combinations(
        range(len_A-1,-1,-1), 2)))

    assert possible_edges.shape == (len_A * (len_A - 1) // 2, 2), possible_edges.shape
    return possible_edges

def _select_n_random_edges(key, possible_edges, n_edges, len_A):
    """
    Select n_edges out of possible_edges without replacement.

    Params:
      key    : the jax.random.PRNGKey
      n_edges: The number of edges to select
      possible_edges: An (m x 2) array of possible edges
                      where m is n * (n - 1) / 2
      len_A  : The length of the adjacency matrix from which
               edges are to be selected.
    Returns:
      i_s, j_s : an array pair of the ith and jth nodes corresponding to the edges
    """

    edges = jax.random.choice(key, possible_edges, shape=(n_edges,), replace=False)

    return edges

def flip_with_prob(key_i, x_i, prob_i):
    u = jax.random.uniform(key_i)
    predicate = u < prob_i
    return jax.lax.cond(predicate, lambda x: flip(x), lambda x: x, x_i) 

def flip(xi, edge_dtype=jnp.int32):
    """
    Flips the value xi. xi is 0 or 1
    Function is intended to be vmapped and jit compiled
    Params:
      xi : an element of the array x
    Returns
      flipp_ed xi : the flipped element
    """

    return jax.lax.cond(xi, lambda xi : edge_dtype(0), lambda xi : edge_dtype(1), xi) 

def flip_edges(key, edge_vector, flip_probs, len_edge_vector):
    """
    jittable
    Params:
      key : a jax PRNGKey
      edge_vector : a vector whose elements are 0 or 1
      len_edge_vector : the length of the edge vector
      flip_prob: a jnp.array of flip_probabilities 
                 

    Return:
      flipped_vector : the updated vector with the flipped elements
    """

    keys = jax.random.split(key, len_edge_vector)
    return jax.vmap(flip_with_prob)(keys, edge_vector, flip_probs)

def flip_adjacency__j(key, A, prob: float, possible_edges, n_edges: int, len_A: int):
    """
    1. Randomly select n_edges from adjacency matrix A
    2. With probability p, flip an edge (if its 0 flip to 1 and vice versa)
    3. Return the updated matrix

    The advantage of this approach for the mover is that
    when constructing the proposal distribution q the evaluation of
    q(xi|xj) is easy as it is prob^n_flipped_edges

    Params:
      key
      A
      prob : the probability of flipping an edge
      possible_edges : an array of possible edges. See get_possible_edges()
      n_edges : int the length of A
    Returns:
      A - an updated adjacency matrix. Note the diagonal is 0 due to the tril implementation
    """
    keys = jax.random.split(key) # (2, 2)
    edge_arr = _select_n_random_edges(keys[0],  possible_edges, n_edges, len_A) # (n_edges, 2)
    i_s = edge_arr[:, 0] # (n_edges,)
    j_s = edge_arr[:, 1] # (n_edges,)
    edges = A[i_s, j_s] # (n_edges,)

    flip_probs = jnp.ones(n_edges) * prob # (n_edges,)
    flipped_edges = flip_edges(keys[1], edges, flip_probs, n_edges) # (n_edges,) 
    A = A.at[i_s, j_s].set(flipped_edges) # (len_A, len_A)
    L = jnp.tril(A, k=-1) # diag are zeros
    A = L + L.T
    return A


def _move_mu__j(key, mean, sigma, n):
    """
    
    """
    return mean + sigma*jax.random.normal(key, shape=(n,)) 

def move_Cs(Ss, ts):

    prey = np.where(Ss >= ts)
    return prey
    


def _move_Sigma_inv(key, V, n, p):
    """
    jittable implementation
    Params:
      p - length of V
      n - wishart degrees of freedom. n > p - 1
    """

    new_Sigma = pynet_rng.wishart(key, V, n, p)
    return new_Sigma






def move_edges(key, A, prob, n_edges: int):
    """
    Sample n edges at a time each with a probability of prob

    Params: 
      A an adjacency matrix
      prob : float the probability of an edge occuring
      shape    : the length of A
      n_edges  : the number of edges to move
      key  : A jax prng key

    Returns:
      A : a new adjacency matrix
    """
    n = len(A)
    assert 2 < n
    assert 0 < n_edges <= n * (n - 1) / 2

    A = _move_edges_j(key, A, prob, n_edges, n) # jittable
    
    return A


def move_model(key, M0, proposal_params: dict):
    """
    Take a step in parameter space
    Params:
      key - jax PRNGKey
      M0  - the model dictionary at the current time step
      proposal_params - configure the proposal distribtuion
    """

    return M1


def plot_sample_space(Ss, textx=8, texty=250):
    n_possible_prey = len(Ss)
    n_k = np.array([int(comb(n_possible_prey, k)) for k in range(1, n_possible_prey + 1)])
    x = np.arange(n_possible_prey) + 1
                
    plt.style.use('ggplot')
    plt.plot(x, n_k, 'b.')
    plt.xlabel('n prey in composite')
    plt.ylabel('n composites')
    plt.title((f"Sample space ("u"\u03A9"
    f") of possible composites"
    f"\n{n_possible_prey} possible prey"))


    text = u'|\u03A9|=' + "{:,}".format(sum(n_k))
    plt.text(textx, texty, text)
    plt.show()

def plot_solution_space(Ss):
    outcome_frequency, score_set = enumerate_Ss(Ss)
    n_solutions = len(set(Ss))
    n_total_outcomes = sum(outcome_frequency)
    solution_probability = outcome_frequency / n_total_outcomes
    plt.style.use('ggplot')
    plt.title('Solution space')
    plt.plot(score_set, 'r.', label='min SAINT score')
    plt.plot(solution_probability, 'b.', label='solution probability')
    plt.xlabel('solution number')
    text = f"n-solutions: {n_solutions}"
    plt.legend()
    plt.text(0, 0.85, text)

def plot_samples(Ss):


    lpmf = log_pdf_Cs__Ss_ts_builder(Ss)

    n_examples = 10000

    key = jax.random.PRNGKey(13)
    k1, k2, k3 = jax.random.split(key, 3)
    ts_unif = jax.random.uniform(k1, shape=(n_examples,))
    ts_norm = 0.6 + jax.random.normal(k2, shape=(n_examples,))*0.09
    ts_60 = jax.random.uniform(k3, minval=0.59, maxval=1.0, shape=(n_examples,))
    uniform_scores = [lpmf(C(Ss, t_unif), Ss) for t_unif in ts_unif]
    normal_scores = [lpmf(C(Ss, t_norm), Ss) for t_norm in ts_norm]
    uniform_60_scores = [lpmf(C(Ss, t_60), Ss) for t_60 in ts_60]
    w = 6
    h = 2
    nbins = 50
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False,
    figsize=(w, h))
    plt.style.use('ggplot')

    plt.subplot(121)
    plt.title('Uniform sampling')
    plt.hist(np.array(ts_unif), bins=nbins)
    plt.subplot(122)
    plt.title('Normal sampling')
    plt.hist(np.array(ts_norm), bins=nbins)
    fig.text(0.5, -0.1, 'SAINT inclusion threshold', ha='center')
    plt.show()

def plot_triple_samples(Ss):
        
    lpmf = log_pdf_Cs__Ss_ts_builder(Ss)

    n_examples = 10000

    key = jax.random.PRNGKey(13)
    k1, k2, k3 = jax.random.split(key, 3)
    ts_unif = jax.random.uniform(k1, shape=(n_examples,))
    ts_norm = 0.6 + jax.random.normal(k2, shape=(n_examples,))*0.09
    ts_60 = jax.random.uniform(k3, minval=0.59, maxval=1.0, shape=(n_examples,))

    uniform_scores = [lpmf(C(Ss, t_unif), Ss) for t_unif in ts_unif]
    normal_scores = [lpmf(C(Ss, t_norm), Ss) for t_norm in ts_norm]
    uniform_60_scores = [lpmf(C(Ss, t_60), Ss) for t_60 in ts_60]
                                                
                                                    
    nbins=50                                                        
    w=10
    h=2
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=False,
    figsize=(w, h))
    plt.style.use('ggplot')
    plt.subplot(131)
    plt.title('Uniform sampling')
    plt.hist(np.array(uniform_scores), bins=nbins)
    plt.subplot(132)
    plt.title('Normal sampling')
    plt.hist(np.array(normal_scores), bins=nbins)
    plt.subplot(133)
    plt.title('Uniform Sampling (0.59, 1.0]')
    plt.hist(np.array(uniform_60_scores), bins=nbins)
    fig.text(0.5, -0.1, 'Composite identity score', ha='center')
    fig.text(0.03, 0.5, 'frequency', va='center', rotation='vertical')
    plt.show()

def plot_score_saint(Ss):
    lpmf = log_pdf_Cs__Ss_ts_builder(Ss)
    ts = np.arange(0, 1, 0.01)
    lpmfs = [lpmf(C(Ss, t), Ss) for t in ts]
    plt.plot(ts, lpmfs)
    plt.xlabel('SAINT threshold')
    plt.ylabel('Composite identity Score')


def plot_As_solution_space(n):
    """
    plot the solution space of an adjacency matrix As
    as a function of the size of the composite (its length) n
    """

    
    # the index is the number of edges
    # the value is the number of solutions
    m = int(n*(n- 1) / 2)
    n_solutions = [comb(m, k) for k in range(0, m + 1)] 
#    assert sum(n_solutions) == 2**m, f"{sum(n_solutions), 2**size_As}"

    plt.title(f"Adjacency matrix solution space\nN={n}")

    plt.style.use('ggplot')
    plt.plot(n_solutions)
    plt.xlabel('n edges')
    plt.ylabel('n adjacency matrices')
    return n_solutions



def proposal(key, n, n_examples, p=0.5):

    A = jax.random.bernoulli(key, p=p, shape=(n, n))
    A = jnp.tril(A, k=-1)
    A = A + A.T
    return A

def nedges(A):
    A = jnp.tril(A, k=-1)
    return jnp.sum(A)


def build_re_mcmc(       
                  n_mcmc_steps: int, 
                  betas, 
                  p_lpdf, 
                  q_lpdf, 
                  score_array_len: int,
                  x, 
                  q_rv, 
                  swap_interval: int,
                  save_every: int):

    """
    Build a replica exchange mcmc kernal
    Kernal inputs - rseed
    outputs       - scores at each replica

    q - the proposal distribution
    p   is the posterior
    
    Params:
      n_mcmc_steps -   total number of moves to make for each chain
      betas        -   an array of inverse temperatures, the length of
                       which determines the number of replicas
      
      p_lpdf           :: (x,) -> ScoreArray
                       the log probability density/mass function. May be unnormalized
      q_lpdf           :: (x,) -> ScoreArray
                       the log probability density/mass function of the proposal. 
                       May be unnormalized
      score_array_len  The score length 
      x                initial coordinates
      q_rv             :: (key, x) -> x
      swap_interval    how often to attempt a swap 
      save_interval    how often to save the coordinates
    Returns:
      re_mcmc_kernal :: (key,) -> MCMC_Results
    """

    
    n_replicas = len(betas)
    assert betas.shape == (n_replicas,), f"betas is not an array of inverse temperatures"
    assert n_replicas > 1, f"N replicas {n_replicas}"
    assert isinstance(n_mcmc_step, int)
    assert isinstance(score_array_len, int)
    assert isinstance(swap_interval, int)

    def mcmc_kernal(key, x, beta):
        """
        The mcmc kernal 
        """
        keys = jax.random.split(key, 2)
        u = jax.random.uniform(keys[0])
    
    def sample_MH(key, x, beta):
        """
        Perfrom a single local step in parameter space according to the
        proposal distribution q. Accept or reject based on the Metropolis Criterion.
        q is assumed to be asymetric q(x1|x0) != q(x0|x1). Therefore the conditional density for q
        is required to account for detailed balance.
        Params:
          key - the jax PRNGKey
          x   - the model variable
          ulpdf the unormalized log probability density funciton
          q     the proposal distribtuion
          q_cond_lpdf 
        """
        keys = jax.random.split(key)
        x1 = q(keys[0], x)
    
        u1_score_array = ulpdf(x1)
        u0_score_array = ulpdf(x)
    
        q0_cond_score_array = q_cond_lpdf(x0, x1)
        q1_cond_score_array = q_cond_lpdf(x1, x0)
    
        alpha = min(1, np.exp(ulpdf(x1) + q_cond_lpdf(x1, x) - ulpdf(x) - q_cond_lpdf(x, x1))) 
        alpha = min(1, np.exp(u1_score_array[0] + q1_cond_score_array[0] - u0_score_array[0] - q0_cond_score_array[0])) 
        u = jax.random.uniform(keys[1])
        accepted = u <= alpha
        if accepted:
            x = x1
        return accepted, x
    def re_mcmc_kernal(rseed):
        """
        The Replica exchange Markov chain Monte Carlo algorithm
        (int,) :: -> MCMC_Results
        Params:
          rseed    - int
        Returns:
          MCMC_Results
        """

        # create an array to save the RE scores accross all temperatures

        scores = jnp.zeros((n_replicas, n_mcmc_steps, score_array_len)) 
        accepted = jnp.zeros((n_replicas, n_mcmc_steps), dtype=bool) 

        n_swaps = n_mcmc_steps // swap_interval
        
        # swapper (0, 1), (t1), (t2)
        swaps_accepted = jnp.zeros((n_swaps, 3))

        for mcmc_step in range(0, n_mcmc_steps):
            if True:
                ...
                # perform RE swap

            else:
                # evolve the single chains
                for r_index in range(n_replicas):
                    ...
                    










    


            
            
            
            
            
            
    n_replicas = len(betas)

    re_score_tensor = jnp.zeros()   # (n_temperatures, n_samples, n_restraints)
    acceptance_ratios = jnp.zeros() # (n_temperatures)
    swaps = jnp.zeros() # (n_samples // swap_interval,  )

    key = jax.random.PRNGKey(rseed)

    
    # Evolve N systems according to MCMC

    # Swap every swap interval

    # cycling the temperatures  

    # observe the scores and keep track of the swaps

    # save the trajectories.


    if k > 0 and k % swap_interval == 0:
        # Attempt RE Swap
        ...

    else:
        # Do local sampling
        ...

    network_model_samples

    

