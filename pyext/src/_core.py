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

from inspect import getmembers, signature

import pyile

# Mutable default args and kwargs
mutable = object()

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
    def __init__(self, namespace_dict=None):
        if namespace_dict is None:
            namespace_dict = {}
        for key, val in namespace_dict.items():
            setattr(self, key, val)


class ModelTemplate:
    """
    1. Define the model position PyTree
      - Define the keys in the position dict
      - The shapes of the leaf types don't matter because that's defined at run time by the initial position


    """

    def __init__(
        self, 
        position: dict = mutable, 
        proposal: dict = mutable, 
        restraints: dict = mutable,
        group_ids: list = mutable,
        do_checks=True
    ):
        if position is mutable:
            position = {}
        if proposal is mutable:
            proposal = {}
        if restraints is mutable:
            restraints = {}
        if group_ids is mutable:
            group_ids = []

        self.position = position  # the keys of integer numbers are reserved for node ids
        self.proposal = proposal
        self.restraints = restraints
        self.group_ids = group_ids
        self.do_checks = do_checks

    def build(self, build_options: dict = mutable, do_checks=True):
        if build_options is mutable:
            build_options = {"position_as_dict": True}

        self.validate_nodes()
        self.validate_params()
        return _build_model(self, do_checks=do_checks, **build_options)

    def add_contiguous_nodes(self, start, stop):
        _assert_fun(start < stop, self.do_checks)
        for i in range(start, stop):
            add_point(self, i, do_checks=self.do_checks)

    def add_restraint(self, scope_key, mapping: dict, logprob_fn, options: dict = mutable):
        """
        Add a restraint to the model template

        Args:
          scope_key : key for the position dict
          mapping :
            scope_name : parameter_name
          restraint : Restraint

        Define the restraint
        append the restraint to the restraint_list
        """
        if options is mutable:
            options = {}
        _add_restraint(self, scope_key, mapping, logprob_fn, self.do_checks, **options)

    def add_node_group(self, indices, init_values):
        add_node_group(self, indices, init_values, self.do_checks)
        self.group_ids.append(indices)

    def help_restraint(self, scope, name="anon"):
        """
        Get help on a restraint specified in a scope 
        """
        #print(f"scope {scope} name {name}")
        help(self.restraints[scope][name])

    def add_point(self, point_name: str, init_value: dict = mutable):
        """
        Add a 'point' to the model position
        Args:
          point_name: str
          init_value: dict

        """
        if init_value is mutable:
            init_value = {}
        add_point(self, point_name, init_value, self.do_checks)


    def validate_nodes(self):
        for node in self.position:
            int_t = isinstance(node, int)
            tup_t = isinstance(node, tuple)
            if tup_t:
                for j in node:
                    assert isinstance(j, int)
            assert (int_t or tup_t), f"{node} failed"

    def validate_params(self):
        for node in self.position:
            for param in self.position[node]:
                assert isinstance(param, str), f"{param} not str"

class ModFileWriter:
    def __init__(self, model_template):
        self.mt = model_template

    def to_modfile(self):
        keys = list(self.mt.position.keys())
        nodes = filter(lambda key: isinstance(key, int), keys)
        nnodes = len(list(nodes))
        _assert_fun((nnodes + len(self.mt.group_ids)) == len(keys), "nodes and groups don't match keys", self.mt.do_checks)

        l1 = f"N    nodes    {nnodes}\n"
        l2 = f"N    group    {len(self.mt.group_ids)}\n"
        l3 = f"N    restr    {len(self.mt.restraints)}\n"
        l4 = f"N    param      \n"
        rlines = self.to_restraint_lines()

        return l1 + l2 + l3 + l4 + rlines

    def to_restraint_lines(self):
        lines = ""
        for scope_key, scope in self.mt.restraints.items(): 
            for name, r in scope.items():
                lines += f"R    {scope_key}    {name}    {r.__name__}\n"    
        return lines
        









# Functions that add things to the model template
def add_point(model_template, point_name: str, init_value: dict=mutable, do_checks=True):
    """Adds a point to the model position"""
    if init_value is mutable:
        init_value = {}
    _add_attribute_to_model_template(model_template, point_name, init_value, do_checks)


def add_node_index(
    model_template, index: VertexIndex, init_value: dict = mutable, do_checks=True
):
    """Adds a node to the model position"""
    if init_value is mutable:
        init_value = {}
    _add_attribute_to_model_template(model_template, index, init_value, do_checks)


def add_node_group(
    model_template, indices: NodeIndices, init_value: dict = mutable, do_checks=True
):
    """Adds a group of nodes to the model"""
    if init_value is mutable:
        init_value = {}
    _add_node_group(model_template, indices, init_value, do_checks=True)


def add_node_indices(
    model_template, indices: NodeIndices, init_values: list = mutable, do_checks=True
):
    """
    Adds multiple node indices to the model which are not already present
    """
    if init_values is mutable:
        init_values = []
    _assert_fun(len(init_values) == len(indices), f"unequal lengths", do_checks)
    _assert_fun(isinstance(indices, tuple), f"indices are not a tuple", do_checks)
    for j, idx in enumerate(indices):
        _assert_fun(isinstance(idx, int), f"index {idx} is not an int", do_checks)
        _add_attribute_to_model_template(model_template, idx, init_values[j], do_checks)


# A node is analgous to an IMP particle
# A vertex has to do with a graph


def _build_mapping_fn(keysA, keysB) -> Jittable:
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
        a: tuple = Atup(**a)
        b: tuple = Btup(*a)
        return b

    return mapping_fn


def get_key_positions(selected_keys: list, d: dict) -> npt.ArrayLike:
    """Return the postion array of selected keys from a dictionary d"""
    key_positions = []
    for i, key in enumerate(d):
        if key in selected_keys:
            key_positions.append(i)
    return np.array(key_positions)


def build_select_dict(selected_keys, d):
    key_positions = get_key_positions(selected_keys, d)
    key_list = list(d.keys())

    selected_dict_def = {key_list[i]: d[key_list[i]] for i in key_positions}


def _map_context_to_kwds(context_to_signature, do_checks=True) -> tuple[str, str]:
    """
    Given a dictionary {'a':'b', 'c':'d'}
    build the parital function signature

    'b=a,d=c'
    """
    s = ""
    context_str = ""
    for key, val in context_to_signature.items():
        _assert_fun(isinstance(key, str), "", do_checks)
        _assert_fun(isinstance(val, str), "", do_checks)
        _assert_fun(key.isalnum(), "", do_checks)
        _assert_fun(val.isalnum(), "", do_checks)
        context_str += f"{key},"

        s += f"{val}={key},"
    s = s.strip(",")
    context_str = context_str.strip(",")
    mapping_str = s
    return context_str, mapping_str


def build_example_mapped(idx):
    def example_mapped(position):
        return example_logprob_fn(position[idx].a, position[idx].b)

    return example_mapped


def _add_group_restraint_to_model(
    model_template, mapping, idxs: NodeIndices, logprob_fn, do_checks=True
):
    """
    Args:
      node_attributes :: A
      mapping :: A -> B
      logprob_fn :: (B) -> float
      restraintId

    Here the logprob_fn is a Jittable function that is only
    a function of the position
    """

    # 1 where should the restraint be defined?
    # 2 Add the necassary parameters to the input context based on the log_density signature
    # 3 Compile the restraint using pyile
    #


def add_node_indices_and_group(
    model_template,
    indices: NodeIndices,
    init_values: list,
    group_init_val: dict=mutable,
    do_checks=True,
):
    if group_init_val is mutable:
        group_init_val = {}
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
    x, attribute, init_value, do_checks=True
):
    """Abstract function for adding pattern"""
    if do_checks:
        _assert_fun(
            attribute not in x.position,
            f"{attribute} already in x.position",
        )
    x.position[attribute] = init_value


def _add_node_group(
    model_template, indices: NodeIndices, init_value: dict = mutable, do_checks=True
):
    if init_value is mutable:
        init_value = {}
    _assert_fun(isinstance(indices, tuple), f"indices are not a tuple", do_checks)
    _assert_fun(
        indices not in model_template.position,
        f"The index group is already in the model position",
        do_checks,
    )
    if do_checks:
        for idx in indices:
            _assert_fun(
                idx in model_template.position, f"node {idx} not in model", do_checks
            )

    _add_attribute_to_model_template(model_template, indices, init_value, do_checks)


def _unscoped_log_density(logprob_fn, default_proposal):

    ...


def update_proposal(model_template, scope_key, param_key, proposal_fn, do_checks=True):
    """
    update the proposal funciton in the proposal dictionary
    """

    if scope_key not in model_template.proposal:
        model_template.proposal[scope_key] = {}

    current = model_template.proposal[scope_key][param_key]

    _assert_fun(
        isinstance(current, type(lambda: ...)),
        "current_fn is not a function type",
        do_checks,
    )
    _assert_fun(
        isinstance(proposal_fn, type(lambda: ...)),
        "proposal_fn is not a function type",
        do_checks,
    )

    model_template.proposal[scope_key][param_key] = proposal_fn


def _add_restraint(
    model_template, 
    scope_key, 
    mapping: dict, 
    logprob_fn, 
    do_checks=True,
    init_position = 1.,
    restraint_base_name="anon",
    auto_rename= False
):
    """
    Add a restraint to the model template
    Add a default mover

    Args:
      scope_key : key for the position dict
      mapping :
        scope_name : parameter_name
      restraint : Restraint

    Define the restraint
    append the restraint to the restraint_list
    """
    print("enter add")
    # 1. If the scope_key in mapping is not in the position dict then the key is added with value init_position 
    for key in mapping:
        if key not in model_template.position[scope_key]:
            model_template.position[scope_key][key] = init_position

    # 2. Build the scoped restraint
    scope_restraint = pyile.build_mapped_fn(mapping, logprob_fn)

    # 3. capture the scope_key by closure
    def restraint(position) -> float:
        return scope_restraint(**position[scope_key])

    mapped_params_doc = ""
    sig = []
    scope_elements = []
    for scope_element, param_element in mapping.items():
        mapped_params_doc += f"  {scope_element} -> {param_element}\n"
        sig.append(param_element)
        scope_elements.append(scope_element)

    docstring = gen_dynamic_docstring(**locals())

    if restraint.__doc__:
        restraint.__doc__ += docstring
    else:
        restraint.__doc__ = docstring

    # Update the model template
    print(model_template, f"model template")
    #model_template.logprob_list.append(restraint)
    updated_dict = model_template.restraints
    if scope_key not in updated_dict:
        updated_dict[scope_key] = {}


    if restraint_base_name in updated_dict[scope_key]:
        if auto_rename:
            b = 1
            newname = restraint_base_name
            while newname in updated_dict:
                newname = f"{restrain_base_name}_{b}"
                b += 1
        else:
            assert False, "{restraint_base_name} already in scope. Change name or set auto_rename=True"
        restraint_base_name = newname
    
    
    updated_dict[scope_key][restraint_base_name] = restraint 


def gen_dynamic_docstring(
    logprob_fn, sig, scope_elements, scope_key, mapped_params_doc, ndashes=26, **kwargs
) -> str:

    docstring = "Dynamic docstring by PyNet\n"
    docstring += "-" * ndashes + "\n"
    docstring += f"""log density : {logprob_fn.__name__}
{scope_key} -> {tuple(scope_elements)} -> {tuple(sig)} -> log_prob\n"""
    docstring += mapped_params_doc
    docstring += f"{logprob_fn.__name__}{signature(logprob_fn)}\n"
    docstring += "-" * ndashes
    return docstring
