"""
Helper functions for the core module used during model definition
"""
from collections import namedtuple


def list_to_string(l: list):
    """
    Returns a string "a b c" from a list [a, b, c]
    """
    l = [str(i) for i in l]
    return " ".join(l)


def dict_to_namedtuple(d: dict, name: str = "MyTuple"):
    MyTuple = namedtuple(name, d)
    return MyTuple(**d)
