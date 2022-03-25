import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from functools import partial
from typing import Callable as f
from typing import Protocol
from .typedefs import Dimension
import collections
                    # variable value -> Array index
n_samples: int      # [1, n_samples] -> [0, n_samples)
n_prey: Dimension         
n_replicates: int
n_interpolating: int


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



def remove_ith_entry(arr, arr_l: Dimension, i: int):
    """Array has at least two elements"""

    
    def i_eq_0(arr, arr_l: Dimension, i: int):
        """ i=0 -> [1, arr_l)"""
        out_arr = jnp.zeros(arr_l -1)
        out_arr = arr[1:arr_l]
        return out_arr

    def i_eq_arr_l(arr, arr_l: Dimension, i: int):
        """ i=arr_l-1 -> [0, arr_l - 1)"""
        out_arr = jnp.zeros(arr_l -1)
        out_arr = arr[0:arr_l - 1]
        return out_arr

    def zero_lt_i_lt_arr_l(arr, arr_l: Dimension, i: int):
        """ 0 < i < arr_l -> [0, i) c [i+1, arr_l) """
        out_arr = jnp.zeros(arr_l -1)
        out_arr = out_arr.at[0:i].set(arr[0:i])
        out_arr = out_arr.at[i:arr_l-1].set(arr[i+1:arr_l])
        return out_arr

    p1 = i == 0
    
    def branch2(arr, arr_l, i):
        p2 = i == arr_l
        out_arr = jax.lax.cond(p2, i_eq_arr_l, zero_lt_i_lt_arr_l, (arr, arr_l, i))
        return out_arr

    out_arr = jax.lax.cond(p1, i_eq_0, branch2, (arr, arr_l, i))
    return out_arr



    


        
    
    # i=0 -> [1, arr_l)
    # i=n-1 -> [0, arr_l - 1)
    # 0<i<arr_l -> [0, i) concat [i+1, n) 

    #     






