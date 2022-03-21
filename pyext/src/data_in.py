"""
Data -> data_in.obj -> model.obj

The data_in module exposes a uniform interface for varied Data input
to be placed into model objects
"""

import itertools
from typing import Iterator

from .typedefs import TsvPath

def read_column_n(filepath: TsvPath, col_num: int) -> Iterator[str]:
    with open(filepath, 'r') as f:
        for line in f:
            yield line.split('\t')[col_n]
