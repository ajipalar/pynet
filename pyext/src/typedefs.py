#import graph_tool as gt
from pathlib import Path
from typing import (
    Any,
    Callable,
    NewType,
    Optional,
    ParamSpec,
    Protocol,
    Sequence,
    Tuple,
    TypeAlias,
    TypeVar,
    Union
)

import typing
import inspect
import types

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True) 
T_contra = TypeVar('T_contra', contravariant=True)


# Paths
AnyPath = NewType('AnyPath', Path)
DirPath = NewType('DirPath', Path)
FilePath = NewType('FilePath', Path)
PlainTextDataPath = NewType('PlainTextDataPath', FilePath)
TsvPath = NewType('TsvPath', PlainTextDataPath)

# Pandas 
Series = Any
AnyCol = NewType('AnyCol', str)
PGGroupCol = NewType('PGGroupCol', AnyCol)
ColName = NewType('ColName', str)



# str Data Types (entries in the excel sheets)
Bait = NewType('Bait', str)
ExcelEntry = NewType('ExcelEntry', str)
Organism = NewType('Organism', str)
UID = NewType('UID', str)  # UniProtID
PreyUID = NewType('PreyUID', UID)
ProteinName = NewType('ProteinName', str)
GeneID = NewType('GeneID', str)  # NCBI Entrez Gene identifier

#Other Types

# Graph types
#G = NewType('G', gt.Graph)

# Functions

#  pure function
#  nnf: non-negative function
#  nnf -> int: Discrete non negative function
#  nnf -> float: continuous non-negative function
#  zf : partition funciton
#  pdf: probability density function
#  pmf: probability mass function
# lpdf: log probability density function
# lpmf: log probability mass function
# ll:   log likelihood
# lp:   log prior
# lpi:  log posterior
# -ll   negative log likelihood
# -lp   negative log prior
# 

R = TypeVar('R')

# jax related
#Array = Any  # implemented as a class so that Array[int] can be typed
class Array:
    def __getitem__(self, idx):
        return Any

f__ = Array() # float using 32 or 64 bit integers 
i__ = Array()

#Array = Any  # implemented as a class so that Array[int] can be typed
RealArray = Array
IntegerArray = Array
Array1d = NewType('Array1d', Array)
DTypeLikeInt = Any
DTypeLikeFloat = Any
DeviceArray = NewType('DeviceArray', Array)
PRNGKeyArray = Any  # Change this to prng.PRNGKeyArray
KeyArray = Any
Index = NewType('Index', int)

Dimension = NewType('Dimension', int)

# More number types to increase readibility

RV = float # random variate
fParam = float
iParam = int
Prob = float  # if x : Prob 0 <= x <= 1
lProb = float # if x : lProb -inf < x <=0


UINT_DTYPES = Any  # TODO prng.UINT_DTYPES

# imp related - Types are not classes
# Math related
Number: TypeAlias = int | float | complex

#Number = NewType('Number', Union[int, float, complex])
PRNGKey = NewType('PRNGKey', Tuple[int, int])
Vector = NewType('Vector', Sequence[Number])
Matrix = NewType('Matrix', Sequence[Sequence[Number]])
CartesianTable = NewType('CartesianTable', Matrix)

# Generic types
State = NewType('State', object)
Output = NewType('Output', object)

# MCMC related
Samples = NewType('Samples', Array)
Weights = NewType('Weights', Array)  # AIS weights
LogWeights = NewType('LogWeights', Array)  # AIS natural log w

JitFunc = Any
PartialF = Any
PDF = Any
lPDF = Any
PMF = Any
lPMF = Any
PureFunc = Any
DataFrame = Any
GenericInvariants = Any
