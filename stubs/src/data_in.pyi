from pyext.src.typedefs import TsvPath as TsvPath
from typing import Iterator

def read_column_n(filepath: TsvPath, col_num: int) -> Iterator[str]: ...
