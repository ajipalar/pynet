"""
The core design of the pynet module 

Objects

  Model
  Score
  Restraint
    get_pdf
    get_lpdf
    get_pdf_and_grad
    get_lpdf_and_grad

  
variables = update_fun(key, variables)

Library
  rng
  pdf

Consider all the inputs and outputs needed from the top level
    model_template:
      state_template
        const_data
        variables
      scoref_template 
        restraint_template

        

    sampling_template
      n_steps : int
      key     : 


    samples = sample(model_template)
    analysis = analyze(samples)

    sample = build_sample(sampling_args)

    model_state = update(model_state)

    update = build_update(update)
    

Functions the model may need

  Score
    get_val : (i, v) -> [x1, x2, ..., xn]

  Restraint
    get_val : (i, v) -> x


# Define
    Info
    Rep
    Score
    Sampling
  Compile
    Run ->
  Analyze Outputs
"""


import numpy as np
import numpy.typing as npt
from collections import namedtuple
from abc import ABC, abstractmethod
from typing import TypeAlias, Union

from inspect import getmembers

# Define classes
Attribute = namedtuple("Attribute", "name val")

# Define types
NodeIndex: TypeAlias = int
NodeIndices: TypeAlias = npt.ArrayLike


# Some helper funcitons


def safe_update(key, val, d):
    """
    Safely update a dictionary
    """
    assert key not in d
    d[key] = val
    return d


def list_to_string(l: list):
    """
    Returns a string "a b c" from a list [a, b, c]
    """
    l = [str(i) for i in l]
    return " ".join(l)


def dict_asnamedtuple(d: dict, name: str = "MyTuple"):
    """
    converts python dictionary to namedtuple
    """
    MyTuple = namedtuple(name, d)
    return MyTuple(**d)  # namedtuple("Name", d)(**d)


def namedtuple_asdict(t: namedtuple):
    return t._asdict()


def _add_to_namedtuple(key, val, t: tuple, name="MyTuple"):
    """
    Make a new named tuple with an added key, value pair
    do runtime checks
    """
    assert isinstance(t, tuple), f"t {t} is not a tuple"
    assert key not in t, f"key {key} already in tuple"
    d = t._asdict()
    d[key] = val
    t = dict_asnamedtuple(d, name=name)
    return t


def get_visiblemembers(obj) -> iter:
    members = iter(getmembers(obj))
    visible = filter(lambda x: x[0][0] != "_", members)
    return visible


# End helper functions


# Abstract classes


class VectorSpace(ABC):
    def __add__(self, o):
        """vector addition"""
        assert False, f"Must override base class"

    def __radd__(self, o):
        """right vector addition"""
        assert False, f"Must override base class"

    def __mul__(self, o):
        """scalar multiplication"""
        assert False, f"Must override base class"

    def __rmul__(self, o):
        """right scalar multiplication"""
        assert False, f"Must override base class"


class CompositionSpace(VectorSpace):
    def __matmul__(self, o):

        assert False, f"Must override base class"

    def __rmatmul__(self, o):
        assert False, f"Must override base class"


class Function(CompositionSpace):
    """
    The composition space has the following operations defined
      - addition
      - scalar multiplication
      - composition
    """

    def __init__(self, f):
        assert isinstance(
            f, type(lambda x: None)
        ), f"f is not a python function"  # make sure the inputs are python functions
        self.f = f

    def __call__(self, *args, **kwargs):
        assert False, "Functions should not be called directly, use .f"

    def __add__(self, o):
        assert type(o) == Function, f"tried to add a Function to a python function"

        def h(*args, **kwargs):
            return self.f(*args, **kwargs) + o.f(*args, **kwargs)

        return Function(h)


class DataStruct:
    def __init__(self, namespace_dict={}):
        for key, val in namespace_dict.items():
            setattr(self, key, val)





class ModelState(DataStruct):
    """
    A container for model state attributes
    Do not place attributes that begin with an underscore
    """

    pass


class Functions(DataStruct):
    """
    A container for model state attributes
    Do not place attributes that begin with an underscore
    """

    pass


class Model:
    """
    Object oriented
    --------------------------------------
    1. Model definition
       a. Define input information
       b. Define representaiton
       c. Define scoring
       d. Define sampling scheme
          i. Define step
          ii. Define sampling method
          iii. Define sampling paramters

    2. Decompose model into a (model_state, {functions}, meta_data) triple
       a. autograd
    --------------------------------------
    Functional

    3. Optionally apply optimizations
       pmap for multiple processors
       vmap to vectorize functions
    4.
    4.

    A python run time environment is used to define the model
    model instance attributes are defined in this environment
    """

    # ModelState = namedtuple("ModelState", "consts variables scores")

    def __init__(self):

        # assert isinstance(model_state, dict), f"model state {model_state} is not a python dictionary"
        # assert isinstance(functions, dict), f"functions {functions} is not a python dictionary"

        # self.model_state = dict_asnamedtuple(model_state, name="ModelState")
        # self.functions = dict_asnamedtuple(functions, name="Functions")

        self.model_state = ModelState()
        self.functions = Functions()  # Overwriting a key will be silent

    def _scope_to_dict(self, scope) -> dict:
        scope_dict = {key: val for key, val in get_visiblemembers(scope)}
        return scope_dict

    def _add_to_model_state_from_pair(self, key, val):
        attr_dict = self._scope_to_dict(self.model_state)
        assert key not in attr_dict, f"{key} already in model_state"
        self.model_state = ModelState(attr_dict | {key: val})

    def _add_to_model_state_from_dict(self, d):
        attr_dict = self._scope_to_dict(self.model_state)
        assert (
            len(set(d).intersection(attr_dict)) == 0
        ), f"d already shares keys with model_state"
        self.model_state = ModelState(attr_dict | d)

    def add_to_model_state(self, keyval: Union[tuple, dict]):
        """
        Adds something to the model state


        Args: keyval is either a tuple (key, value) pair or
              a dictionary of {key1: val1, ..., keyn: valn}
              The model_state cannot already have key
        """

        if isinstance(keyval, tuple):
            assert len(keyval) == 2, f"keyval needs exactly two entires"
            self._add_to_model_state_from_pair(keyval[0], keyval[1])
        elif isinstance(keyval, dict):
            assert len(keyval) > 0
            self._add_to_model_state_from_dict(keyval)
        else:
            assert False, f"keyval type {type(keyval)} is neither a dict nor a tuple"


    def add_to_functions(self, key, val):
        ...

    def _to_functional(self):
        """
        The model state is a PyTree of data
        The Functions are a PyTree of pure functions that know how to read the model state
        """
        return self.model_state, self.functions


class Restraint:
    """
    Restraint is a template interface that all restraints must follow
    Every Restraint is defined over certain group of node indices, for a certain model

    Evaluating a restraint requires
      - reading parameters and data from the model_state
      - passing those parameters to a scoring function
      - computing the result
      - placing the result in the model_state.scores

    Building a restraint requires
      - naming the restraint with a unique identifier
      - checking that the unique name doesn't already exist
      - defining the node indeces ascociated with the restraint
      - defining the variables  ascociated with the restraint
      - defining the scoring container ascociated with the restraint

    Args:
      name : a unique str identifier
    """

    def __init__(self, name: str, idxs: NodeIndices, model: Model):
        self.name = name
        self.idxs = idxs


def _check_restraint_name(restraint: Restraint, model: Model):
    assert restraint.name not in model.restraints


def add_restraint(restraint: Restraint, model: Model, do_checks=True):

    _check_restraint_name(restraint, model) if do_checks else None


class RestraintAdder:
    """
    Adds a restraint to a model

    Does some checking
    """

    def __init__(self, restraint: Restraint, model: Model):
        self.restraint = restraint
        self.model = model

    def check_name_is_unique(self):
        assert self.model
        
def update_datastruct(ds: DataStruct, keyval: Union[tuple, dict]):
    """
    Adds something to the model state


    Args: keyval is either a tuple (key, value) pair or
          a dictionary of {key1: val1, ..., keyn: valn}
          The model_state cannot already have key
    """
    if isinstance(keyval, tuple):
        assert len(keyval) == 2, f"keyval needs exactly two entires"
        return update_datastruct_from_pair(keyval[0], keyval[1])
    elif isinstance(keyval, dict):
        assert len(keyval) > 0
        return update_datastruct_from_dict(ds, keyval)
    else:
        assert False, f"keyval type {type(keyval)} is neither a dict nor a tuple"

def update_datastruct_from_pair(ds: DataStruct, key, val) -> DataStruct:
    attr_dict = datastruct_to_dict(ds) 
    assert key not in attr_dict, f"{key} already in datastruct"
    return DataStruct(attr_dict | {key: val})

def update_datastruct_from_dict(ds: DataStruct, d: dict):
    attr_dict = datastruct_to_dict(ds) 
    assert (
        len(set(d).intersection(attr_dict)) == 0
    ), f"d already shares keys with data struct"
    return DataStruct(attr_dict | d)

def datastruct_to_dict(ds : DataStruct):
    return dict(get_visiblemembers(ds))

def datastruct_to_tuple(ds: DataStruct, typename: str="MyTuple") -> tuple:
    """
    Converts a datastruct to a namedtuple
    """

    attr_dict = datastruct_to_tuple(ds)
    MyTuple = namedtuple(typename, attr_dict) 
    return MyTuple(**attr_dict)

def tuple_to_datastruct(t: tuple) -> DataStruct:
    """
    Converts a (named)tuple into a datastruct
    """
    tdict = namedtuple_asdict(t)
    return DataStruct(tdict)


