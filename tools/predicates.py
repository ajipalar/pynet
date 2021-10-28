"""
Predicates used for filtering
A predicate is a function that returns a bool
Used with filter and iterators
e.g., filter(predicate, iterator)
"""

from pathlib import Path
from net.typedefs import AnyPath, FilePath, DirPath

def isfile(p: AnyPath) -> bool:
    return p.is_file()

def isxlsx(p: FilePath) -> bool:
    return p.suffix == '.xlsx'

def isjson(p: FilePath) -> bool:
    return p.suffix == '.json'
