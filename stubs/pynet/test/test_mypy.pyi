from typing import Any, Callable, TypeVar

MyConcreteType: Any
MyConcreteDerivedType: Any
MyGenericType = TypeVar('MyGenericType')
T = TypeVar('T')
R = TypeVar('R')
P: Any
my_func2: Callable[..., R]

def my_func(a: MyConcreteType, b: MyConcreteType) -> MyConcreteType: ...
def my_func3(f: Callable[P, R]) -> Callable[P, R]: ...

my_concrete_type: MyConcreteType
world: Any
my_derived_type: MyConcreteDerivedType
x: float
z: int
t: int
y: str
f: str
h: int
