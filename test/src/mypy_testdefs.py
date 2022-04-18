try:
    from IMP.pynet import T as T, T_co as T_co, T_contra as T_contra
    import IMP.pynet.distributions as dist
except ModuleNotFoundError:
    from pyext.src.typedefs import T as T, T_co as T_co, T_contra as T_contra
    import pyext.src.distributions as dist


def generic_function(x: T) -> T:
    return x


dist.norm.pdf(10)
