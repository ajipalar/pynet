"""
Consume search triplets to produce useful search algorithms
"""
import collections
import functools
from functools import partial
import inspect
import itertools
import operator
import os
from typing import Any, Callable, Optional, Sequence, Set, Tuple, TypeVar, List


import jax
import jax.numpy as jnp
import jax.scipy as jsp
import _search_producers as producers
from collections import namedtuple


def metropolis_hastings(x, f, g, rseed):
    """Get the metropolis hastings algorithm for f(x), rseed, and transition distribution g 

    Args:
      x: independant pytree variable
      f: scoreing functions such that score = f(x)
      g: transition distribution such that x = g(key, x)
    Returns:
      move: int -> ss -> ss 
      sample: int -> ss
    Examples:

    """
    search_init, search_update, get_x = producers.metropolis_hastings(x, f, g, rseed)

    def move(step, search_state):
        score = f(get_x(search_state))
        search_state = search_update(step, score, search_state)
        return search_state

    return move

def gibbs(x, cond_lst, g, rseed):

    search_init, search_update, get_x = producers.gibbs(x, cond_lst, rseed, move_conditional)

    def move(step, search_state):
        

        return search_state

    return move
    

def sample(n_steps, move):
    search_state = search_init(x)
    search_state = jax.lax.fori_loop(
        lower=0, upper=n_steps, body_fun=move, init_val=search_state
    )
    return search_state





