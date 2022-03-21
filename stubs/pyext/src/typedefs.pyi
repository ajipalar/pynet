from typing import Any, TypeAlias, TypeVar

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)
T_contra = TypeVar('T_contra', contravariant=True)
AnyPath: Any
DirPath: Any
FilePath: Any
PlainTextDataPath: Any
TsvPath: Any
Series = Any
AnyCol: Any
PGGroupCol: Any
ColName: Any
Bait: Any
ExcelEntry: Any
Organism: Any
UID: Any
PreyUID: Any
ProteinName: Any
GeneID: Any
R = TypeVar('R')

class Array:
    def __getitem__(self, idx): ...

f__: Any
i__: Any
RealArray = Array
IntegerArray = Array
Array1d: Any
DTypeLikeInt = Any
DTypeLikeFloat = Any
DeviceArray: Any
PRNGKeyArray = Any
KeyArray = Any
Index: Any
Dimension: Any
RV = float
fParam = float
iParam = int
Prob = float
lProb = float
UINT_DTYPES = Any
Number: TypeAlias
PRNGKey: Any
Vector: Any
Matrix: Any
CartesianTable: Any
State: Any
Output: Any
Samples: Any
Weights: Any
LogWeights: Any
JitFunc = Any
PartialF = Any
PDF = Any
lPDF = Any
PMF = Any
lPMF = Any
PureFunc = Any
DataFrame = Any
GenericInvariants = Any
