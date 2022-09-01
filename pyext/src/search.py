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
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from jax.tree_util import Partial
from typing import Callable, TypeVar, Generic

S = TypeVar("S")  # Sample state type
M = TypeVar("M")  # Model type
Y = TypeVar("Y")  # Output type
F = Callable  # jax pure function type
T = Callable  # jax function transformation type


class PyTree(Generic[S]):
    ...


x: PyTree[M]  # The independant variable to be searched
y: PyTree[Y]  # The variable dependant on f and its transformations such as grad
score: float
step: int


f: F  # the scoring function such that score = f(x)
t: T  # a function transformation t

search_state: PyTree[S]
search_init: F  #  search_state = search_init(x)
search_update: F  #  search_state = search_update(step, y, search_state)
get_x: F  # x = get_x(search_state)


step: F

# Producer of Search Triple


def get_mh(x, f, g, rseed):
    """
    x: dependant variable
    f: scoreing function
    g: transition distribution
    rseed: int random seed
    """
    SearchState = namedtuple("SS", "x y k")
    k_start = jax.random.PRNGKey(rseed)

    def search_init(x0):
        return SearchState(x0, f(x0), k_start)

    def get_x(search_state):
        return search_state.x

    def search_update(step, y, search_state):
        k_step, k1, k2 = jax.random.split(search_state.k, 3)
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


# Consumers of Search Triple


def move(step, search_state):
    score = f(get_x(search_state))
    search_state = search_update(step, score, search_state)
    return search_state


def metropolis_hastings(n_steps):
    search_state = search_init(x)
    search_state = jax.lax.fori_loop(
        lower=0, upper=n_steps, body_fun=move, init_val=search_state
    )
    return search_state


# +

example_seed = 13
x_shape = (100, 100)
x = jax.random.normal(jax.random.PRNGKey(example_seed), shape=x_shape)


def f(x):
    return jnp.sum(jsp.stats.norm.pdf(x, loc=1000, scale=99))


gvar = 1000


def g(k, x):
    return jax.random.normal(k, shape=x_shape) * gvar


rseed = 44
search_init, search_update, get_x = get_mh(x, f, g, rseed)

x0 = x
# mh = jax.jit(metropolis_hastings)

# search_state = mh(1000)


def mh(nsteps):
    ss = search_init(x)
    for i in range(nsteps):
        ss = jax.jit(move)(i, ss)
        if i % 1000 == 0:
            print(f"step {i} score {ss.y}")


mh(50000)


# -


# ?jax.lax.cond

# +
def add_two(f, x):
    return f(*x) + 2


f = jsp.stats.bernoulli.pmf
print(f"Normal invocation {add_two(jsp.stats.bernoulli.pmf, (1, 2))}")
try:
    jax.jit(add_two)(f, (1, 2))
    print("jit should not succeed")
except:
    print("jit failed")

f = jax.tree_util.Partial(f)

print(f"jit success! {jax.jit(add_two)(f, (1, 2))}")

# -

f

# +
search_state = {"params": 0, "prev_step": 0}


def get_params(search_state):
    return search_state["params"]


def init_fun(x):
    return {"params": x, "prev_step": 100, "score": s(x), "score0": s()}


def update_fun(step, search_state):

    ...


class Search:
    score: float
    s: Callable
    ...


# -

# # Sampling and Optimization Using Jax Transformations
#
# After defining a model representation `x` and a scoring function `f` we mush search over the possible values `x` may take using some search scheme. This search may consist of enumeration, filtering, optimization and sampling. It can be very useful to tailor such searches to custom modeling problems, therefore it is desirbale to have a convient way of expressing such searches while being a general as possible.
#
# Here we propose the following abstractions inspired by experimetal jax optimizers library.
# We would like a convient abstraction for creating custome Monte Carlo searchs easily using
# arbitrary models and scoring functions.
#
# To achieve this and use Jax's function transformations we constrain our models `x` to be a jax type. That is, `x` must be a PyTree. Formally
#
# - let `x` be some independant variable of interest to be searched.
# - let the type of `x` be a PyTree whose structure is invariant over the course of the search
# - let `f` be a pure function such that `score = f(x)` where `score` could be the
#   result from evaluating a log probability density or loss function
# - let `t` be a Jax function transformation such that the `t(f)` preserves `f`'s signature.
#   e.g. `f_prime = grad(f)` where `grad` is a function transformation.
# - let `y` be some variable that depends on `t(f)(x)`. For any transformation `t`
# - let `step` index the number of steps the search has taken
#
#
# A search procedure must do the following
#   1) get some information about the state of the model `x` using the scoring function `f`.
#
#       This could simply be the `score` at the current and previous `step` as is the case for the Metropolis     Hastings Algorithm, or it could be the gradient of `f(x)` as is the case for stochastic gradient descent.
#
#   2) update the model `x` according to some rules including backpropagation or Monte Carlo searches
#
#   3)
# - get some information about the model `x` using the scoring function
# - update the model
# - update `x` in some way either using gradients or by evaluating `f(x)`
#
# >hello this
# is indentent text
# as is this
# >
#
#

from collections import namedtuple

ValueAndGrad = namedtuple("ValueAndGrad", "val grad")
val_and_grad = ValueAndGrad(1.0, 2.0)

from jax.tree_util import tree_flatten, tree_unflatten

flat, tree = tree_flatten(val_and_grad)

v2 = tree_unflatten(tree, flat)

tree_flatten(jnp.array(2))

# +
# Goals

# x: PyTree of Jax types.

# score = s(x)


# Implement Gibbs, MH, and Slice sampling using the triplet of functions model


def mh(k, f, nsteps):
    k = jax.random.split(k, nsteps)
    for i in range(nsteps):

        ratio = lpdf(x) - lpdf(x0)
        ratio = jnp.exp(ratio)
        u = jax.random.uniform(k[i])

        if u <= ratio:
            x = x0

    return x
