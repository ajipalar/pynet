from .typedefs import DeviceArray, PRNGKeyArray, T, T_co, T_contra 
from typing import Union, Protocol

class SupportsPDF(Protocol[T_contra]):
    @staticmethod
    def pdf(x: T_contra) -> float: ...

class SupportsLogPDF(Protocol[T_co]):
    @staticmethod
    def lpdf(x: T_contra)  -> float: ...

class SupportsRV(Protocol[T_co]):
    @staticmethod
    def rv(key: PRNGKeyArray) -> DeviceArray: ...

class SupportsScore(Protocol):
    @staticmethod
    def score(*args, **kwargs) -> float: ...

class SupportsLogScore(Protocol):
    @staticmethod
    def lscore(*args, **kwargs) -> float: ...

