"""
Predicates used for filtering
A predicate is a function that returns a bool
Used with filter and iterators
e.g., filter(predicate, iterator)
"""

from pathlib import Path
from .typedefs import AnyPath, FilePath, DirPath
import numpy.typing as npt
import numpy as np


def isfile(p: AnyPath) -> bool:
    return p.is_file()


def is_suffix_T(p: FilePath, T: str) -> bool:
    """Returns true if the file has the suffix T
    T: 'xlsx', 'json', 'xls'
    """
    return p.suffix == f".{T}"


def isxlsx(p: FilePath) -> bool:
    return p.suffix == ".xlsx"


def isexcel(p: FilePath) -> bool:
    return (p.suffix == ".xlsx") or (p.suffix == ".xls")


def isjson(p: FilePath) -> bool:
    return p.suffix == ".json"


# linalg predicates


def is_vector(a: npt.NDArray) -> bool:
    return True if (is_row_vector(a) or is_column_vector(a)) else False


def is_row_vector(a: npt.NDArray) -> bool:
    return True if (a.ndim == 2) else False


def is_column_vector(a: npt.NDArray) -> bool:
    return False if (len(a.shape != 2) or (a.shape[1] != 1)) else True


def is_square_matrix(a: npt.NDArray) -> bool:
    return False if ((a.shape[0] != a.shape[1]) or (len(a.shape) != 2)) else True


def is_symmetric_square_matrix(a: npt.NDArray):
    return True if (is_square_matrix(a) and np.all(a == a.T)) else False


def is_lower_tri(a: npt.NDArray) -> bool:
    return True if (is_square_matrix(a) and np.all(a == np.tril(a))) else False


def is_upper_tri(a: npt.NDArray) -> bool:
    return True if (is_square_matrix(a) and np.all(a == np.triu(a))) else False


def is_nonnegative(a: npt.NDArray) -> bool:
    return False if np.any(a < 0) else True


# Array tests
def is_array1d(a: npt.NDArray) -> bool:
    return a.ndim == 1


def is_array2d(a: npt.NDArray) -> bool:
    return a.ndim == 2


def is_array_square(a: npt.NDArray) -> bool:
    return True if ((a.ndim == 2) and (a.shape[0] == a.shape[1])) else False
