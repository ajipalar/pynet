import numpy as np
from jax import jit, grad, vmap
import jax.numpy as jnp
import jax; jax.config.update('jax_platform_name', 'cpu')
import timeit


def myfunc(x, y):
    return x**y

jx_myfunc = jit(myfunc)
a = 10.1
b = 11.2
n = 1000000

def f2():
    for i in range(1000):
        x = np.arange(50)
        y = x**2
        return y

jx_f2 = jit(f2)
if __name__ == "__main__":
    def f1():
        myfunc(a, b)
    def j1():
        jx_myfunc(a, b)
    import timeit
    print(timeit.timeit(f1, number=n))
    print(timeit.timeit(j1, number=n))
    print(timeit.timeit(f2, number=n))
    print(timeit.timeit(jx_f2, number=n))
