import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.tree_util import tree_flatten, tree_unflatten
from collections import namedtuple

"""
The following factories produce search triplets 
    All triplets have optional args and kwargs (a -> args -> kwargs -> b) which are ommited for readability
    (search_init :: (key -> x -> ss)
     search_update :: (step -> y -> ss)
     get_x :: (ss -> x)
    )

    factories define the type ss
    factories define the type y


Provides search triplet producers. Not a public API. see search.py

 X: PyTree def for x
 S: PyTree def for search_state
 Y: PyTree def for y

 search_init : (X) -> S
 search_update : (int, *, *) -> *
 get_x : (*) -> *

 score = f(x, *args, **kwargs)

 To be as general as possible searches require the following

 - An initial configuration for x
 - A random starting seed
 - A scoring function f(x)
 - The definition of the search scheme
 - Some hyper parameters
   - including the number of steps to take
   - the transition distribution
   - and others

searches will consume the following function triplet
- search_init
  - initial configuration of x
  - starting

 search_init
     - initial configuration for x
     - a random seed
     - 

"""

def metropolis_hastings(f, g):
    """
    Metroplis Hasting triplet factory
      - search_state = search_init(x)
      - search_state = search_update(step, y, search_state)
      - x = get_x(search_state)


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
    def search_init(key, x):
        """The initial configuration for x is given by x
        key is always unconsumed"""
        return SearchState(x, f(x), key)

    def get_x(search_state):
        return search_state.x

    def search_update(step, y, search_state):
            k_step, k1, k2 = jax.random.split(search_state.key, 3)
            x_step = g(k1, get_x(search_state))
            y_step = f(x_step)
            u_step = jax.random.uniform(k2)

            search_state = jax.lax.cond(
                pred=u_step < y_step,
                true_fun=lambda : SearchState(x_step, y_step, k_step),
                false_fun=lambda : SearchState(search_state.x, search_state.y, k_step),
            )
            return search_state

    return search_init, search_update, get_x


def gibbs(rseed, draw_conditional, n_conditionals : int):
    """
    A factory for gibbs sampling
    

    Args:
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

    def search_init(x):
        return SearchState(x, key)


    def search_update(step, y, search_state):
        keys = jax.random.split(search_state.key, self.pre_split_keys + 1)

        def body(i, val):
            x, keys = val
            x_i = self.draw_conditional(keys[i], x, i)
            x = set_x
                
    def get_x(search_state):
        return search_state.x

    return search_init, search_update, get_x


def log_monte_carlo(x):

    SearchState = namedtuple("SearchState", "x key")

    return search_init, search_update, get_x
