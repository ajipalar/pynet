"""Unix command line alias to use in python"""
import os
import pathlib
from pathlib import Path, PosixPath

class CL(object):
    """An alias for clear in python"""
    def __repr__(self):
        os.system("clear")
        return ""

class PWD(object):
    """An alias for pwd in python."""
    def __repr__(self):
        return os.getcwd()

class LS(object):
    """An alias for ls in python."""
    def __repr__(self):
        cwd: PosixPath = Path.cwd() 
        for path in cwd.iterdir(): print(path)
        return ""



cl = CL()
pwd = PWD()
ls = LS()
