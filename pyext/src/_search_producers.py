import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.tree_util import tree_flatten, tree_unflatten
from collections import namedtuple

"""
Provides search triplet producers. Not a public API. see search.py

 X: PyTree def for x
 S: PyTree def for search_state
 Y: PyTree def for y

 search_init : (X) -> S
 search_update : (int, *, *) -> *
 get_x : (*) -> *

 score = f(x, *args, **kwargs)

"""


class MetropolisHastings:

    SearchState = namedtuple("SearchState", "x y key")

    def __init__(self, f, g, rseed):
        self.f = f
        self.g = g
        self.rseed = rseed
        self.k_start = jax.random.PRNGKey(self.rseed)

    def __call__(self):
        def search_init(x):
            return MetropolisHastings.SearchState(x, self.f(x), self.k_start)

        def get_x(search_state):
            return search_state.x

        def search_update(step, y, search_state):
            k_step, k1, k2 = jax.random.split(search_state.key, 3)
            x = g(k1, get_x(search_state))
            u = jax.random.uniform(k2)

            x = jax.lax.cond(
                pred=u < search_state.y,
                true_fun=lambda x: x,
                false_fun=lambda x: search_state.x,
                operand=x,
            )
            return SearchState(x, f(x), k_step)

        return search_init, search_update, get_x

def metropolis_hastings(f, g, rseed):
    """
    Get the triplet for the metropolis hastings algorithim.

    Args:
      x: dependant variable
      f: scoring function.
      g: transition distribution. x = g(key, x)
      rseed: int random seed

    Returns:
      search_init : (*) -> *
      search_update : (int, *, *) -> *
      get_x : (*) -> *

    Examples
    """
    SearchState = namedtuple("SearchState", "x y key")
    key = jax.random.PRNGKey(rseed)
    def search_init(x):
        return SearchState(x, f(x), key)

    def get_x(search_state):
        return search_state.x

    def search_update(step, y, search_state):
            k_step, k1, k2 = jax.random.split(search_state.key, 3)
            x = g(k1, get_x(search_state))
            u = jax.random.uniform(k2)

            x = jax.lax.cond(
                pred=u < search_state.y,
                true_fun=lambda x: x,
                false_fun=lambda x: search_state.x,
                operand=x,
            )
            return SearchState(x, f(x), k_step)

    return search_init, search_update, get_x



class Gibbs:
    """
    Perform Gibbs sampling on the model

    Args:
      x :: Tree
        The dependant argument, some pytree
      f :: Tree -> int  -> score
        A conditional
      rseed :: int
        A random seed
      draw_conditional ::
        draw_conditional(i, x, search_state)

    Returns:

    Examples:

    Types:
      Tree : A generic pytree type
      Tree[A] : A specific pytree structure type
      *    : A wildcard representing a leaf. Leaves can be arrays and literals
      Flat : [*, ...] A generic flattened PyTree. A list of leaves
      Flat[Tree] : A flattened PyTree type corresponding to the structure of Tree
      TreeDef : A generic tree metadata type
      TreeDef[Tree] : Tree metadata corresponding to Tree

      x :: Tree[X]
      f :: Tree[X] -> int -> score
      rseed :: int
      search_state :: Tree[S]
      key :: PRNGKey
      draw_conditional :: PRNGKey, int -> (Tree[X] -> int -> score) -> *


    Notes:

      We need a list of conditional distributions or a single conditional distributions

      draw_conditional (key, x, i) -> x_i
      
      

    """

    SearchState = namedtuple("SearchState", "x key")

    def __init__(self, x, rseed, draw_conditional, n_conditionals=None, pre_split_keys=False):
        self.x = x
        self.rseed = rseed
        self.draw_conditional = draw_conditional
        self.k_start = jax.random.PRNGKey(self.rseed)

        if not n_conditionals:
            n_conditionals = len(x)

        self.n_conditionals = n_conditionals

    def __call__(self):
        def search_init(x):
            return Gibbs.SearchState(x, self.key)


        if self.pre_split_keys:
            def search_update(step, y, search_state):
                keys = jax.random.split(search_state.key, self.pre_split_keys + 1)

                def body(i, val):
                    x, keys = val
                    x_i = self.draw_conditional(keys[i], x, i)
                    x = set_x
                    
        def get_x(search_state):
            return search_state.x

        return search_init, search_update, get_x


def gibbs(x, f, rseed, draw_conditional):
    return Gibbs(x, f, rseed, draw_conditional)()

"""
def metropolis_hastings(f, g, rseed):
    return MetropolisHastings(f, g, rseed)()
"""
