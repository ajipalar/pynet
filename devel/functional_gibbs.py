#!/usr/bin/env python
# coding: utf-8

# In[250]:


import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.lax import fori_loop as jfori
from functools import partial
import pyext.src.plot as plot
from typing import Any
import termplotlib as tpl
State = Any
Output = Any
KeyArray = Any
DeviceArray = Any

class StateLess:
    def stateless_method(state: State, *args, **kwargs) -> (State, Output):
        pass

# Funcitonal implementation
def cond_norm(key, y, rho, *args, **kwargs):
    return jax.random.normal(key) * jnp.sqrt(1 - rho **2) + y * rho

def cond_dist(key, dist, params, **kwargs):
    return dist(key, *params, **kwargs)


# X and Y are independant
def gibbsf(key: KeyArray, *args, N=10, thin=10, rho=-0.3, **kwargs) -> DeviceArray:
    
    def outer_body_fun(i: int, val):
        key, samples, x, y, rho, thin  = val
        key, samples, x, y, rho, thin = jax.lax.fori_loop(0, thin, inner_body_fun, val)
        samples = samples.at[i].set([x, y])
        return key, samples, x, y, rho, thin

    def inner_body_fun(i: int, 
                       init_val
                       ) -> tuple[KeyArray, float, float]:
        
        key, samples, x, y, rho, thin = init_val
        key, k1, k2 = jax.random.split(key, 3)
        x = cond_dist(k1, cond_norm, (y, rho), **kwargs)
        y = cond_dist(k2, cond_norm, (x, rho), **kwargs)
        return key, samples, x, y, rho, thin
    
    # initiate
    key, k1, k2 = jax.random.split(key, 3)
    x = jax.random.uniform(k1)
    y = jax.random.uniform(k2)
    samples = jnp.zeros((N, 2))
    val = key, samples, x, y, rho, thin
    
    # key, samples, x, y, rho, thin
    samples = jax.lax.fori_loop(0, N, outer_body_fun, val)[1]
    return samples

# imperative implementation
def gibbs(key, N=10, thin=10):
    key, k1, k2 = jax.random.split(key, 3)
    x = jax.random.uniform(k1)
    y = jax.random.uniform(k2)
    samples = jnp.array(np.zeros((N, 2)))
    for i in range(N):
        for j in range(thin):
            key, k1, k2 = jax.random.split(key, 3)
            x = x_cond_dens(k1, y)
            y = y_cond_dens(k2, x)
        samples = samples.at[i].set([x, y])
    return samples

gibbsf_partial = partial(gibbsf, N=5000, thin=100)
gibbsf_jit = jax.jit(gibbsf_partial, static_argnames='rho')
key = jax.random.PRNGKey(5)
samples = np.array(gibbsf_jit(key))
x, y = samples[:, 0], samples[:, 1]
tpl.plot(x, y)
tpl.show()




gibbsf(key)


# In[259]:


samples = gibbsf_jit(key, rho=0)
plot.scatter(samples[:, 0], samples[:, 1])


# In[23]:


plot.marginal(x, xlabel='x')


# In[24]:


plot.marginal(y, xlabel='y')


# In[229]:


# Metropolis Hastings


def mh_f(key, steps):
    
    def score(x):
        return jax.scipy.stats.logistic.pdf(x)

    def rv(key):
        return jax.random.uniform(key, minval=-10, maxval=10)

    def body(i, init):
        key, x = init
        x0 = x[i-1]
        key, g1, g2 = jax.random.split(key, 3)
        x1 = rv(g1)
        A = score(x1) / score(x0)
        p = jax.random.uniform(g2)
        xi = jax.lax.cond(p < A, lambda : x1, lambda : x0)
        x = x.at[i].set(xi)
        return key, x
    
    def silly(i, v):
        u, x = v
        w = u, x
        return w
    
    samples = jnp.zeros(steps)
    key, samples = jax.lax.fori_loop(1, steps, body, (key, samples))
    return samples

key = jax.random.PRNGKey(10)
key, j1 = jax.random.split(key, 2)
steps = 100000
mh_f(key, steps)


# In[230]:


get_ipython().run_line_magic('timeit', 'mh_f(key, steps)')


# In[232]:


mh_fp = partial(mh_f, steps=100000)


# In[236]:


mh_f_p_jit = jax.jit(mh_fp)


# In[237]:


get_ipython().run_line_magic('timeit', 'mh_f_p_jit(key).block_until_ready()')


# In[241]:


jnp.all(mh_fp(key) == mh_f_p_jit(key))


# In[248]:


x = np.array(mh_f_p_jit(key))
plot.marginal(x)


# In[249]:


s = np.array(score(x))
plot.marginal(s)


# In[247]:





# In[39]:


np.quantile(x, np.arange(0, 1, 0.1))


# In[35]:


np.arange(0, 1, 0.1)


# In[4]:


k1 = jax.random.PRNGKey(23)
k2 = jax.random.PRNGKey(23)


# In[11]:


get_ipython().run_line_magic('timeit', 'gibbsf_jit(k1).block_until_ready()')


# In[13]:


get_ipython().run_line_magic('timeit', 'gibbs(k2, N=5000, thin=100)')


# In[5]:


def rel_error(x, s):
    return s / x

def error():
    r1 = rel_error(0.160, 0.0177)
    r2 = rel_error(59.9, 0.0373)
    r3 = r1 + r2
    return (59.9 / 0.160 ) * r3


# jit:    160 ms ± 17.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# python: 59.9 s ± 37.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 
# jit compilation appeared to accelerate the gibbs sampler by 374 ± 42

# In[9]:


jax.make_jaxpr(f)


# In[ ]:


"""
# This code will take a very long time to jit compile
# because the gibbs function uses python for loops
# instead of jax.lax.fori_loop

gibbs_partial = partial(gibbs, N=50000, thin=1000)
gibbs_jit = jax.jit(gibbs_partial)
key = jax.random.PRNGKey(5)
samples = np.array(gibbs_jit(key))
plot.scatter(samples[:, 0], samples[:, 1])
"""


# In[3]:


key = jax.random.PRNGKey(5)
gibbs_jit = jax.jit(gibbs)
samples = gibbs_jit(key)


# In[4]:


key = jax.random.PRNGKey(5)
gibbsf_jit = jax.jit(gibbsf)
samplesf = gibbsf_jit(key)

# samples == samplesf
# In[7]:


key2 = jax.random.PRNGKey(17)
samples = gibbs_jit(key2)
key2 = jax.random.PRNGKey(17)
samplesf = gibbsf_jit(key2)


# In[13]:


plt.scatter(samplesf[:, 0], samplesf[:, 1])


# In[4]:


gibbs_partial = partial(gibbsf, N=50000, thin=1000)
gibbsf_jit = jax.jit(gibbs_partial)
key = jax.random.PRNGKey(5)
samples = np.array(gibbsf_jit(key))
plt.scatter(samples[:, 0], samples[:, 1])


# In[14]:


x, y, z, *args = (1, 2, 3)


# In[ ]:


try:  # Doesn't work; fori_loop requires f(i, val)
    f = lambda x: x+1
    jax.lax.fori_loop(0, 100, f, 0)
except TypeError:
    # Works
    # Can jit through the funciton
    def iwrapper(f):
        def wrap(i, val):
            return f(val)
        return wrap
    f = iwrapper(lambda x: x+1)
    print(jax.lax.fori_loop(0, 10, f, 0))
    print(jax.jit(lambda : jax.lax.fori_loop(0, 10, f, 0))().block_until_ready())


# In[270]:


def f(x: KeyArray):
    def g(x):
        return 2*x
    return g(x) + 1
f = jax.jit(f)


# In[80]:


def inner_body_fun(i: int, 
                   val: tuple[KeyArray, float, float]
                   ) -> tuple[KeyArray, float, float]:

    key, samples, x, y, thin = val
    key, k1, k2 = jax.random.split(key, 3)
    x = x_cond_dens(k1, y)
    y = y_cond_dens(k2, x)
    samples = samples.at[i].set([x, y])
    return key, samples, x, y, thin

key = jax.random.PRNGKey(10)
key, k1, k2 = jax.random.split(key, 3)
N = 1000
thin = 5
x = jax.random.uniform(k1)
y = jax.random.uniform(k2)
samples = jnp.zeros((N, 2))
val = key, samples, x, y, thin
def outer(val):
    return jax.lax.fori_loop(0, N, inner_body_fun, val)

opartial = partial(outer, val=val)
otest = jax.jit(opartial)


# In[94]:


a, b, *args = (1, 2, 3)


# In[185]:


def f1(x, y, *args, **kwargs):
    return x * y, args, kwargs

def f2(x, *args, **kwargs):
    rho = kwargs['rho']
    return x + rho, args, kwargs

def g(x, y, *args, **kwargs):
    a, args, kwargs = f1(x, y, *args, **kwargs)
    b, args, kwargs = f2(a, *args, **kwargs)
    return b, args, kwargs

def h(x, y, *args, **kwargs):
    return f2(f1(*args, **kwargs))
g(1, 2, **{'rho': 4})


# In[100]:


# Suppose we have a set of functions a, b, c, ..., h
def local_block():
    def a(state: tuple[Any], x, y, *args, **kwargs):
        return state, (x + y, )

    def b(state: tuple[Any], x, *args, **kwargs):
        return state, (x, x ** 2 )

    def c(state: tuple[Any], x, y, *args, **kwargs):
        return state, (x + y, ), args, kwargs

    def d(state: tuple[Any], x, rho, *args, **kwargs):
        return state, (x - 10 * rho,)
    
    def e(state: tuple[Any], x, *args, **kwargs):
        return state, (x * 2, )
    
    def f(state: tuple[Any], x, *args, **kwargs):
        return state, (jnp.log(x), )
    
    def h(state: tuple[Any], x, *args, **kwargs):
        return state, (1 / 2 * x ** 2, )
    
    # We wish to compose and jit compile these functions into a pipe
    # output = h(e(d(c(b(a)))))
    # each function has positional[:tuple] and keyword[:dict] arguments
    # we would like to make a change in the middle of the chain without
    # affecting how we built the funciton before and after
    # how would we do this?
    
    # supposing the funcitons are pure
    
    # we must make a change to c
    # option1 : add an extra parameter to the highest level function
    # and a parameter to c without changing c's return type
    
    # c(x, y) -> c(x, y, z)
    # 
    
    def statless_function(state: State, *args, **kwargs) -> tuple[State, Output]:
        pass
    
    def pipe(state: tuple[Any], pipe_in, *args, **kwargs):
        state, return_a = a(state, *pipe_in)
        state, return_b = b(state, *return_a)
        state, return_c = c(state, *return_b)
        state, return_d = d(state, *return_c)
        state, return_e = e(state, *return_d)
        state, return_f = f(state, *return_e)
        state, pipe_out = h(state, *return_f)
        return state, pipe_out
    
    class P:
        def __lt__(self, div):
            print(div)
    
    state = (1, 2)
    pipe_in = (11, 12)
    p = P()
    p < pipe(state, pipe_in)
    #p < jax.make_jaxpr(a)((), np.array([1, 2]), np.array([1, 1]))
    p < jax.make_jaxpr(pipe)((1, 2), (11, 12))
    
        
local_block()


# In[107]:


def f(x, y, z, *args, file=None, **kwargs):
    print(file)
    return x, (y + z, ), args, kwargs

state, ret, args, kwargs = f(*(1, 2, 3, 4, 5), **{'j': 2, 'file':4})
state, ret, args, kwargs


# In[119]:


def f(state, *args, **kwargs):
    print(args)
    x = args[0]
    print(x)
    print(args)
    y = args[1]
    return args


def f(state, x, y, *args, **kwargs):
    ret_f = x + y
    return state_f, ret_f, args_f, kwargs_f


def producer(*args):
    return 1, 2

def middle(x, y, *args):
    print(x + y)
    print(args)
    return x + y, args
    
def consumer(z, *args):
    z = z**2
    print(z)
    print(args)
    return z



consumer(*middle(*producer()))


# In[23]:


def f1(x, y, *args, **kwargs):
    return x * y, args, kwargs

def f2(x, *args, **kwargs):
    return x + 2
print(jax.make_jaxpr(f2)b)


# In[24]:


jxp_f2 = jax.make_jaxpr(f2)


# In[25]:


jxp_f2


# In[175]:


g(1, 2, **{'rho': 10})


# In[176]:


def f(x, y, **kwargs):
    return x * y * rho


# In[180]:


f(1, 2, **{'rho':10})


# In[195]:


m[9]

