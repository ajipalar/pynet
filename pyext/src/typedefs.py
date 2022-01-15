import graph_tool as gt
import pandas as pd
from pathlib import Path
from typing import Any, Callable, NewType, ParamSpec, TypeVar
import typing
import inspect
import types

T = TypeVar('T')
P = ParamSpec('P')

DataFrame = NewType('DataFrame', pd.DataFrame)
DirPath = NewType('DirPath', Path)
FilePath = NewType('FilePath', Path)
AnyPath = NewType('AnyPath', Path)
PlainTextDataPath = NewType('PlainTextDataPath', FilePath)
TsvPath = NewType('TsvPath', PlainTextDataPath)

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

#Functions
Function = NewType('Function', Callable[[T], P])

PureFunction = NewType('PureFunction', Function)
Parameters = NewType('Parameters', 
             dict[str, inspect.Parameter])

