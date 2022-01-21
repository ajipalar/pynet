#import graph_tool as gt
import pandas as pd
from pathlib import Path
from typing import (
        Any, 
        Callable, 
        NewType, 
        Optional,
        ParamSpec, 
        Sequence, 
        Tuple,
        TypeVar, 
        Union
)
import typing
import inspect
import types

T = TypeVar('T')

#Paths
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
#G = NewType('G', gt.Graph)

#Functions
P = ParamSpec('P')
R = TypeVar('R')

#jax related
Array = Any
RealArray = Array
IntegerArray = Array
DTypeLikeInt = Any
DTypeLikeFloat = Any
PRNGKeyArray = Any #Change this to prng.PRNGKeyArray
KeyArray = NewType('KeyArray', PRNGKeyArray) 

UINT_DTYPES = Any #TODO prng.UINT_DTYPES

#imp related - Types are not classes
#Math related
#Number = NewType('Number', Union[int, float, complex])
Number = Union[int, float, complex]
PRNGKey = NewType('PRNGKey', Tuple[int, int])
Vector = NewType('Vector', Sequence[Number])
Matrix = NewType('Matrix', Sequence[Sequence[Number]])

