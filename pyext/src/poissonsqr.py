import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from functools import partial
from typing import Callable as f
from typing import Protocol
from .typedefs import Dimension, JitFunc
import collections
                    # variable value -> Array index
n_samples: int      # [1, n_samples] -> [0, n_samples)
n_prey: Dimension         
n_replicates: int
n_interpolating: int

pynet_jax_fdtype = jnp.float32 

# samples, weights = model.sample.ais(n_samples, n_interpolating)

PoissonSQR = collections.namedtuple(
    "PoissonSQR", ["param", "data"])

SQRParam = collections.namedtuple(
    "SQRParam", ["eta1", "eta2", "theta", "Phi"])

AIS = collections.namedtuple(
    "AIS", ["T", "source", "target", "intermediate"])

Source = collections.namedtuple(
    "Source", ["rv"])

Get = collections.namedtuple(
    "Get", ['eta1', 'eta2'])




get = Get(
    lambda phi, i: phi[i, i], 
    lambda theta, phi, x, i: theta[i] + 2)


def remove_ith_entry__s(arr_l: Dimension) -> JitFunc:
    i_eq_0__j = i_eq_0__s(arr_l)
    i_eq_arr_l__j = i_eq_arr_l__s(arr_l)
    zero_lt_i_lt_arr_l__j = zero_lt_i_lt_arr_l__s(arr_l)
    branch2__j = branch2__s(arr_l)
    
    def remove_ith_entry__j(arr, i):
        out_arr = jax.lax.cond(
            i==0,
            lambda a, b: i_eq_0__j(arr=a),
            lambda a, b: branch2__j(arr=a, i=b),
            *(arr, i))
        return out_arr

    return remove_ith_entry__j



def remove_ith_entry(arr, arr_l: Dimension, i: int):
    """Array has at least two elements"""
    out_arr = jax.lax.cond(i==0, lambda a, b, c: i_eq_0(a, b), branch2, *(arr, arr_l, i))
    return out_arr
    
#def i_eq_0(arr, arr_l: Dimension, i: int):
def i_eq_0(arr, arr_l): 
    """ i=0 -> [1, arr_l)"""
    out_arr = jnp.zeros(arr_l -1, dtype=pynet_jax_fdtype)
    out_arr = arr[1:arr_l]
    return out_arr

def i_eq_0__s(arr_l: Dimension):
    i_eq_0__j = partial(i_eq_0, arr_l=arr_l)
    return i_eq_0__j

#def i_eq_arr_l(arr, arr_l: Dimension, i: int):
def i_eq_arr_l(arr, arr_l):
    """ i=arr_l-1 -> [0, arr_l - 1)"""
    out_arr = jnp.zeros(arr_l -1, dtype=pynet_jax_fdtype)
    out_arr = arr[0:arr_l - 1]
    return out_arr

def i_eq_arr_l__s(arr_l: Dimension):
    i_eq_arr_l__j = partial(i_eq_arr_l, arr_l=arr_l)
    return i_eq_arr_l__j



#def zero_lt_i_lt_arr_l(arr, arr_l: Dimension, i: int):
def zero_lt_i_lt_arr_l(arr, arr_l, i):
    """ 0 < i < arr_l -> [0, i) c [i+1, arr_l) """
    out_arr = jnp.zeros(arr_l-1, dtype=pynet_jax_fdtype)
    # out_arr = out_arr.at[0:i].set(arr[0:i])

    # Not differentiable, bounds unknown at trace time
    out_arr, tmp  = jax.lax.fori_loop(
        0, i,
        lambda j, t: (t[0].at[j].set(t[1][j]), t[1]),  
        (out_arr, arr)) 

    out_arr, tmp  = jax.lax.fori_loop(
        i+1, arr_l,
        lambda j, t: (t[0].at[j-1].set(t[1][j]), t[1]),  (out_arr, arr)) 
        
    return out_arr

def zero_lt_i_lt_arr_l__s(arr_l):
    return partial(zero_lt_i_lt_arr_l, arr_l=arr_l)


#def branch2(arr, arr_l, i):
def branch2(arr, arr_l, i):
    out_arr = jax.lax.cond(i==arr_l-1, lambda a, b, c: i_eq_arr_l(a, b), zero_lt_i_lt_arr_l, *(arr, arr_l, i))
    return out_arr

def branch2__s(arr_l: Dimension):
    i_eq_arr_l__j = i_eq_arr_l__s(arr_l)
    zero_lt_i_lt_arr_l__j = zero_lt_i_lt_arr_l__s(arr_l)
    def branch2__j(arr, i):
        return jax.lax.cond(
            i==arr_l-1,
            lambda a, b : i_eq_arr_l__j(arr=a),
            lambda a, b : zero_lt_i_lt_arr_l__j(arr=a, i=b),
            *(arr, i))

    return branch2__j
    



    


        
    
    # i=0 -> [1, arr_l)
    # i=n-1 -> [0, arr_l - 1)
    # 0<i<arr_l -> [0, i) concat [i+1, n) 

    #     






