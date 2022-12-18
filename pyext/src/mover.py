"""
Attention: The mover functions are meant to be jit compiled using JAX
Without jit compilation the behavior of movers are undefined.

In Jax:
    d = p; d['x'] = 2
    does not change p
In Python:
    It does
    assert d['x'] == p['x']


The movers module defines movers with the following siganture
A distribution takes in a key and params and outputs params
x = rng.norm(key, x)

Movers must read from the model position and write to the model position
Therefore they must be built

To build a mover we must
- know the model position
- know how to map the input parameters to the output
- know the rng
- know the sample state.

sample_state1 = step(sample_state0)
By Defining movers we are defining the proposal distribution.


"""
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import pynet_rng as rng

class Build():
    """
    Builds a mover

    
    """
    def __init__(self):
        ...

def build(gid: str, vid: str, rng):
    """
    Args:
      gid - group id
      pid - variable id
      rng - random number generator function
    """

    def move(key, position):
        new_param = rng(key, state)
        position[gid][vid] = new_param
        return position
    return move





