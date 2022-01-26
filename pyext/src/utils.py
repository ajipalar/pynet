import inspect
from pathlib import Path
import sys

pynet_home = Path('.').absolute().parent
sys.path.append(str(pynet_home))

try:
    import IMP.pynet.config as config
except ModuleNotFoundError:
    import pyext.src.config as config


def modparse(local_dict):
    modname = local_dict['__name__']
    if "." in modname:
        modname = modname.split(".")[1]
    packname = local_dict['__package__']
    return modname, packname


def moduleinfo(local_dict):
    modname, packname = modparse(local_dict)
    print(f'[Package, Module]: {packname} {modname}')


def doc(x):
    """
    Print the docstring
    """
    print(x.__doc__)


def psrc(x):
    print(inspect.getsource(x))


def ls(p:Path):
    for f in p.iterdir():
        print(str(f))


if config.PRINT_MODULE_INFO:
    moduleinfo(locals())
