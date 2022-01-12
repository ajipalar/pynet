import typing as t
from typing import NewType
import pandas as pd
from pathlib import Path
import graph_tool as gt

DataFrame = NewType('DataFrame', pd.DataFrame)
DirPath = NewType('DirPath', Path)
FilePath = NewType('FilePath', Path)
AnyPath = NewType('AnyPath', Path)

#Column label types
AnyCol = NewType('AnyCol', str)
PGGroupCol = NewType('PGGroupCol', AnyCol)

#Data Types (entries in the excel sheets)
ExcelEntry = NewType('ExcelEntry', str)
UID = NewType('UID', str) #UniProtID
PreyUID = NewType('PreyUID', UID)
Bait = NewType('Bait', str)
Organism = NewType('Organism', str)

#Graph types
G = NewType('G', gt.Graph)
