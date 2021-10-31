import typing as t
from typing import NewType
import pandas as pd
from pathlib import Path

DataFrame = NewType('DataFrame', pd.DataFrame)
DirPath = NewType('DirPath', Path)
FilePath = NewType('FilePath', Path)
AnyPath = NewType('AnyPath', Path)

#Column label types
AnyCol = NewType('AnyCol', str)
PGGroupCol = NewType('PGGroupCol', AnyCol)

#Data Types (entries in the excel sheets)
UID = NewType('UID', str) #UniProtID

