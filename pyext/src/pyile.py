"""
Pyile:
    A mini python to python compiler implemented in python
"""
import numpy as np
from inspect import signature


def deff(fname):
    return f"def {fname}("


def siglist(l=[], code=""):
    for j, i in enumerate(l):
        if j > 0:
            code += f" {i},"
        else:
            code += f"{i},"
    return code


def pos_only(positional_only_arglist: list[str], code=""):
    return siglist(l=positional_only_arglist, code=code)


def varargs(code="", stargs="*args"):
    return code + stargs + ","


def kw_or_pos(code="", kw_or_pos: list[str] = []):
    return siglist(l=kw_or_pos, code=code)


def kw_only(code="", kwds_only: list[str] = []):
    return siglist(l=kwds_only, code=code)


def var_kw(code="", vk="**kwargs"):
    return code + " " + vk


def rparen(code):
    return code + "):"


def rparaen_and_anno(code, anno):
    return code + f") -> {anno}:"


def i4(code):
    """
    indent four spaces
    """
    return "    " + code


def i4s(code):
    c = ""
    for line in code.split("\n"):
        c += i4(line)
    return c


def assemble_gsig(kwds_only):
    code = ""
    fname = "pyile_anon"
    code = deff(fname)
    code = varargs(code, stargs="*")
    code = kw_only(code, kwds_only)
    code = var_kw(code)
    code = rparen(code)
    return code


def generate_params(f):
    return iter(param for _, param in signature(f).parameters.items())


def generate_position_only_parameters(it):
    return filter(lambda x: x.kind == x.POSITIONAL_ONLY, it)


def position_only_params_from_f(f):
    a = generate_position_only_parameters
    b = generate_params
    return a(b(f))


def smap(mapping: dict, f, do_checks=True):
    """
    signature map

    if f::S -> Y
    and mapping is dict[a:s]
    get the string representation of g
    g::A -> Y

    g accepts arguments in A by keyword defined in mapping
    g passes arguments to f by position

    do_checks:
      check_all arguments are mapped
    
    """
    inverse_mapping = {param: key for key, param in mapping.items()}
    fname = f.__name__

    keys_a = [key for key in mapping]
    parameters = [mapping[key] for key in mapping]

    positional_only = list(position_only_params_from_f(f))
    positional_only_names = [x.name for x in positional_only]

    if do_checks:
        assert len(mapping) == len(inverse_mapping), "the map and inverse map are not one-to-one"
        assert len(mapping) == len(positional_only_names), f"mapping and pos params don't match {mapping, positional_only_names}"
        assert len(inverse_mapping) == len(positional_only_names), "inv mapping and pos params don't match"
        assert set(positional_only_names) == set(inverse_mapping), f"inv mapping and params don't match"

    new_ordering = [inverse_mapping[param] for param in positional_only_names]

    code = assemble_gsig(keys_a)

    body = f"return {fname}("
    body += pos_only(new_ordering)
    body = body.strip(",")
    body += ")"
    code = code + "\n" + i4s(body)

    return code

def build_mapped_fn(mapping, f):
    """
    
    """
    code = smap(mapping, f)
    g = {f.__name__: f}
    l = dict()
    exec(code, g, l)
    return l['pyile_anon']


"""
def g(*, a, b, c, **kwargs):
    return f(b, c, a):
"""
