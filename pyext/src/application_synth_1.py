import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import _core

m = _core.ModelTemplate()
m.add_group('root', (0, 1, 2, 3), init_adjacency=True)

@jax.jit
def f(x):
    return x

position = f(m.position)

A = position['root']['A']
pi = position['root']['pi_'] 
A[pi]

if __name__ == "__main__":
    print("Done!")
