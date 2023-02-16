"""
A numpy compatible model prototype not focused on performance
"""

import numpy as np
import scipy as sp
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from collections import deque
from itertools import combinations, chain

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











