import jax
import jax.numpy as jnp
import jax.scipy as jsp
from collections import namedtuple

"""
Provides search triplet producers. Not a public API. see search.py

 X: PyTree def for x
 S: PyTree def for search_state
 Y: PyTree def for y

 search_init : X -> S
 search_update : (int, *, *) -> *
 get_x : (*) -> *

"""


def metropolis_hastings(x, f, g, rseed):
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
    k_start = jax.random.PRNGKey(rseed)

    def search_init(x0):
        return SearchState(x0, f(x0), k_start)

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
