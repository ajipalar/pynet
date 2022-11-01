"""
Search provides funcitonality for searching parameters spaces.
Including Monte Carlo, Optimizaiton, and Filtering.

This module is modeled in part after the Jax.experimental_libraries.optimizers module

Definitions used throughout the module

  Jax Types are PyTrees.
  x: independant vairable. Must be a PyTree
  f: a pure scoring function
  score: score = f(x)
  t: a function transformation that leaves the function signature unchanged
  y: a dependant variable. Must be a PyTree.
     could be dependant on f(x) and/or t(f)(x)

  
  A search triple is a triple of the following pure functions

    search_state = search_init(x)
    search_state = search_update(step, y, search_state)
    x = get_x(search_state)

  A search_producer is some callable object that returns
  a search triple.


  A search consumer is takes a search triple as input and
  produces some callable (often jittable) object such as a sampling algorithm.
     
     
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
from ._search_producers import metropolis_hastings
