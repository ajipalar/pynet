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
        assert isinstance(f, type(lambda x: None)) , f"f is not a python function"# make sure the inputs are python functions 
        self.f = f

    def __call__(self, *args, **kwargs):
        assert False, "Functions should not be called directly, use .f"

    def __add__(self, o):
        assert type(o) == Function, f"tried to add a Function to a python function"
        def h(*args, **kwargs):
            return self.f(*args, **kwargs) + o.f(*args, **kwargs) 
        return Function(h)



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


    def __init__(self, model_state : dict =  {}, functions : dict = {}):
        assert isinstance(model_state, dict), f"model state {model_state} is not a python dictionary"
        assert isinstance(functions, dict), f"functions {functions} is not a python dictionary"

        self.model_state = dict_asnamedtuple(model_state, name="ModelState")
        self.functions = dict_asnamedtuple(functions, name="Functions")

    def add_to_model_state(self, key, val):
        ...

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


