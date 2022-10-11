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
from typing import TypeAlias, Union, Callable, Any

from inspect import getmembers

# Define classes
Attribute = namedtuple("Attribute", "name val")

# Define types
VertexIndex: TypeAlias = int
NodeId: TypeAlias = (
    Any  # some hashable identifer, for vertices the node index is the vertex index
)
NodeIndices: TypeAlias = npt.ArrayLike
Jittable: TypeAlias = Callable  # A jittiable funciton
PyTree: TypeAlias = Any  # A jax PyTree
NodeAttributes: TypeAlias = dict
Node: TypeAlias = tuple[NodeId, NodeAttributes]


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


class UID(str):
    """
    A globally (to the model) unique id.
    The unique id is used to specific getters and setters for the model attributes in a functional style
    """


class GID(str):
    """A group id for a group of node indices"""

    ...


class CGID(GID):
    """A groupd of ordered contiguous node indices in ascending order"""

    ...


class DataStruct:
    def __init__(self, namespace_dict={}):
        for key, val in namespace_dict.items():
            setattr(self, key, val)


class ModelTemplate:
    def __init__(self, position: dict = {}):
        self.position = (
            position  # the keys of integer numbers are reserved for node ids
        )

    def build(self, build_options: dict = {"position_as_dict": True}, do_checks=True):
        return _build_model(self, do_checks=do_checks, **build_options)


# Functions that add things to the model template
def add_point(model_template, point_name: str, init_value=0, do_checks=True):
    """Adds a point to the model position"""
    _add_attribute_to_model_template(model_template, point_name, init_value, do_checks)


def add_node_index(
    model_template, index: VertexIndex, init_value: dict = {}, do_checks=True
):
    """Adds a node to the model position"""
    _add_attribute_to_model_template(model_template, index, init_value, do_checks)


def add_node_group(
    model_template, indices: NodeIndices, init_value: dict = {}, do_checks=True
):
    """Adds a group of nodes to the model"""
    _add_node_group(model_template, indices, init_value, do_checks=True)


def add_node_indices(
    model_template, indices: NodeIndices, init_values: list, do_checks=True
):
    """
    Adds multiple node indices to the model which are not already present
    """
    _assert_fun(len(init_values) == len(indices), f"unequal lengths", do_checks)
    _assert_fun(isinstance(indices, tuple), f"indices are not a tuple", do_checks)
    for j, idx in enumerate(indices):
        _assert_fun(isinstance(idx, int), f"index {idx} is not an int", do_checks)
        _add_attribute_to_model_template(model_template, idx, init_values[j], do_checks)


# A node is analgous to an IMP particle
# A vertex has to do with a graph


def build_mapping_fn(keysA, keysB) -> Jittable:
    """
    Build a function that maps from A to B

    1. Capture the ordering by closure

    (A -> X)
    (X -> B) namedtuple
    (B -> C)
    
    """
    A: TypeAlias = Any
    B: TypeAlias = Any
    C: TypeAlias = Any
    X: TypeAlias = Any

    Atup = namedtuple("A", keysA)
    Btup = namedtuple("B", keysB)

    Mapp = namedtuple("Mapp", keysB)  # the keys are in Y

    def mapping_fn(a: tuple[A]) -> tuple[B]:
        a = Atup(**a) 
        b = Btup(*a)
        return b

    return mapping_fn


def example_logprob_fn(a, b):
    return jnp.log(a) + jnp.log(b)

def build_example_mapped(idx):
    def example_mapped(position):
        return example_logprob_fn(position[idx].a, position[idx].b)

    return example_mapped

def add_restraint_to_model(model_template, idxs: NodeIndices, log_prob):
    """
    Args:
      node_attributes :: A
      mapping :: A -> B
      logprob_fn :: (B) -> float
      restraintId
    """
    ...

def add_node_indices_and_group(
    model_template,
    indices: NodeIndices,
    init_values: list,
    group_init_val={},
    do_checks=True,
):
    add_node_indices(model_template, indices, init_values, do_checks=do_checks)
    add_node_group(model_template, indices, group_init_val, do_checks=do_checks)


def add_singleton_log_density(model_template, idx: NodeId, log_pdf):
    """
    A singleton restraint depends only on the properties of a single node
    """
    # must ascociate the log_pdf with the  idx
    def restraint(position):
        return log_pdf(**position[idx])

    ...


# have log_pdf(args, kwargs):
#      log_pdf(**position)


def add_pair_restraint(model_template):
    """
    A pair restraint depends only on the properties of a pair of nodes
    """


def add_group_restraint(model_template):
    """
    A group restraint depends only on the attributes of a group of nodes.
    A group consists of more than two nodes
    """


class ModelData:
    ...


class ModelState(DataStruct):
    """
    A container for model state attributes
    Do not place attributes that begin with an underscore

    key
    groups : groupIds
    restraints
      groupdIds : restraintId

    restraint parameters





    """

    pass


class Functions(DataStruct):
    """
    A container for model state attributes
    Do not place attributes that begin with an underscore
    """

    pass


class ContiguousGroupAttribute:
    """

    Offest:
      If the NodeIndices are [9, 10, 11]
      then the offset is -9 such that
      NodeIndices + offset = [0, 1, 2] maps to the group attribute
    """

    def __init__(self, offset: int, attribute):
        self.offset = offset
        self.attribute = attribute


def get_value_from_nodeidxs(model_state, idxs: NodeIndices, uid: UID):
    """

    For a given model, get the value of the unique id at the node indices
    """
    return model_state.getters["uid"](idxs)


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

    def __init__(
        self, getters: dict[UID, Jittable] = {}, setters: dict[UID, Jittable] = {}
    ):

        # assert isinstance(model_state, dict), f"model state {model_state} is not a python dictionary"
        # assert isinstance(functions, dict), f"functions {functions} is not a python dictionary"

        # self.model_state = dict_asnamedtuple(model_state, name="ModelState")
        # self.functions = dict_asnamedtuple(functions, name="Functions")

        self.model_state = ModelState()
        self.getters = getters
        self.setters = setters
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


def datastruct_to_dict(ds: DataStruct):
    return dict(get_visiblemembers(ds))


def datastruct_to_tuple(ds: DataStruct, typename: str = "MyTuple") -> tuple:
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


def _assert_fun(pred, msg: str, do_checks=True):
    if do_checks:
        assert pred, msg


def _build_model(model_template, do_checks, position_as_dict):
    if position_as_dict:
        _assert_fun(
            isinstance(model_template.position, dict),
            "position is not a dict",
            do_checks=do_checks,
        )
        return model_template.position
    else:
        assert False, "Model failed to build"


def _add_attribute_to_model_template(
    model_template, attribute, init_value, do_checks=True
):
    """Abstract function for adding pattern"""
    if do_checks:
        _assert_fun(
            attribute not in model_template.position,
            f"{attribute} already in model_template.position",
        )
    model_template.position[attribute] = init_value


def _add_node_group(
    model_template, indices: NodeIndices, init_value: dict = {}, do_checks=True
):
    _assert_fun(isinstance(indices, tuple), f"indices are not a tuple", do_checks)
    _assert_fun(
        indices not in model_template.position,
        f"The index group is already in the model position",
        do_checks,
    )
    if do_checks:
        for idx in indices:
            _assert_fun(
                idx in model_template.position, f"node {idx} node in model", do_checks
            )

    _add_attribute_to_model_template(model_template, indices, init_value, do_checks)
