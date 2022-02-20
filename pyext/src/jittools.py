import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable

def is_jittable(f: Callable, fargs, fkwargs):
    """Is the pure function f jittable with *fargs and **fkwargs? 
       jax.jit(f)(*fargs, **kwargs)"""

    jitf = jax.jit(f)
    jittable = False
    try:
        r = jitf(*fargs, **fkwargs)  
        jittable = True
    except:
        pass
    return jittable

