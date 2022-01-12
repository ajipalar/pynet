import os

class CL(object):
    """An alias for clear in python"""
    def __repr__(self):
        os.system("clear")
        return ""

class PWD(object):
    """An alias for pwd in python."""
    def __repr__(self):
        return os.getcwd()

cl = CL()
pwd = PWD()
