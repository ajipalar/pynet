from inspect import signature
from typing import Any, Callable

from .typedefs import Function, Parameters


def parameters(f: Function):
    return signature(f).parameters


def Domain(f: Function) -> Parameters:
    return signature(f).parameters


def Codomain(f: Function):
    return reveal_type(signature(f).return_annotation)


def composable(f: Function) -> set[Function]:
    pass


y = Domain(Codomain)
reveal_type(y)
