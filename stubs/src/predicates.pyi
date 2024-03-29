from pathlib import Path as Path
from pyext.src.typedefs import AnyPath as AnyPath, DirPath as DirPath, FilePath as FilePath

def isfile(p: AnyPath) -> bool: ...
def is_suffix_T(p: FilePath, T: str) -> bool: ...
def isxlsx(p: FilePath) -> bool: ...
def isexcel(p: FilePath) -> bool: ...
def isjson(p: FilePath) -> bool: ...
