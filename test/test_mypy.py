from typing import Any, Callable, Iterable, Tuple, NewType, TypeVar, ParamSpec

#Conrete type declartions

MyConcreteType = NewType('MyConcreteType', str)
MyConcreteDerivedType = NewType('MyConcreteDerivedType', MyConcreteType)

#Generic type declaration
MyGenericType = TypeVar('MyGenericType')
T=TypeVar('T')
R=TypeVar('R')
P=ParamSpec('P')
#Function declartion
my_func2: Callable[... ,R] #PEP 612

#Function definition
def my_func(a: MyConcreteType, b: MyConcreteType) -> MyConcreteType:
    return MyConcreteType(a + b)

def my_func3(f: Callable[P, R]) -> Callable[P, R]:
    return f(f(*args, **kwargs))

#variable declaration and instantation
my_concrete_type: MyConcreteType = MyConcreteType("Hello ")
world = MyConcreteType("world")
my_derived_type: MyConcreteDerivedType = MyConcreteDerivedType(world)

#Function calls
my_func(my_concrete_type, my_concrete_type)
my_func(my_concrete_type, my_derived_type)
my_func(my_derived_type, my_derived_type)


