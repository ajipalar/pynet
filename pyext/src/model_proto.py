"""
A numpy compatible model prototype not focused on performance
"""

import numpy as np
import scipy as sp
from scipy.special import comb
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from collections import deque
from itertools import combinations, chain
import matplotlib.pyplot as plt
from functools import partial

import lpdf
import pynet_rng



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

def C(Ss, ts):
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
    Cs = np.zeros(n_prey + 1, dtype=int)
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

def _move_edges_j(key, A, prob, n_edges, n):
    """
    jittable implementation - see move_edges
    Params:
      n - length of A
    """
    A = jnp.array(A)
    keys = jax.random.split(key, 3)
    j_s = jax.random.randint(keys[0], shape=(n_edges,), minval=0, maxval=n)
    i_s = jax.random.randint(keys[1], shape=(n_edges,), minval=j_s + 1, maxval=n) 
    vals = jax.random.bernoulli(keys[2], shape=(n_edges,), p=prob)
    A = A.at[i_s, j_s].set(vals)
    L = jnp.tril(A, k=-1)
    A = L + L.T
    return A

def _move_mu__j(key, mean, sigma, n):
    """
    
    """
    return mean + sigma*jax.random.normal(key, shape=(n,)) 

def move_Cs(Ss, ts):

    prey = np.where(Ss >= ts)
    return prey
    


def _move_Sigma_inv(key, Sigma_inv_s, n):
    """
    jittable implementation
    Params:
      n - length of Sigma_inv_s
    """

    new_Sigma = pynet_rng.wishart(key, V, n, p)





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








