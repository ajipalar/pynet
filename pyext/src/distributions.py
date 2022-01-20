import jax
import jax.numpy as jnp
import numpy as np
from abc import ABC
from typedefs import KeyArray

class Distribution(ABC):
    def __init__(self):
        pass

@abstractmethod
    def logpdf(self):
        pass

@abstractmethod
    def pdf(self):
        pass

@abstractmethod
    def logcdf(self):
        pass

@abstractmethod
    def cdf(self):
        pass

@abstractmethod
    def logpmf(self):
        pass

@abstractmethod
    def pmf(self):
        pass

@abstractmethod
    def ppf(self):
        pass

@abstractmethod
    def rv(self, key: KeyArray):
        pass

@abstractmethod
    def sf(self):
        pass

@abstractmethod
    def isf(self):
        pass

@abstractmethod
    def __call__(self, *args):
        """Calls the pdf or pmf function"""
        pass


    



