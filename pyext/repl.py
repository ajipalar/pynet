import os

class cl(object):
    """An alias for clear in python"""
    def __repr__(self):
        exec('os.system("clear")')
        return ""

cl = cl()
