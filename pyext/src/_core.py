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
from functools import partial
from enum import Enum, unique

from inspect import getmembers, signature
import pyile
import lpdf
import ulpdf

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

@unique
class R(Enum):
    """
    Defines a controlled set of restraints, ascociating names with
    a restraints form number
    """
    mvn = 0 # x, mean, cov params
    wsh = 1 # S, nu, V

class RParams(Enum):
    """
    The members of the Enum have parameter names
    that match the signatures of the parameters in the lpdf and lpmf modules
    They are ordered identically
    """
    mvn = ('x', 'mean', 'cov')
    wsh = ('S', 'nu'  , 'V'  )

@unique
class Rinfo(Enum):
    """
    name numeric n_params *r_params
    """
    mvn = (R.mvn.value, len(RParams.mvn.value), RParams.mvn.value) 
    wsh = (R.wsh.value, len(RParams.wsh.value), RParams.wsh.value)


RestraintNumberToBaseName = {form.value:name for name, form in R.__members__.items()} 
RestraintBaseNameToNumber = {val: key for key, val in RestraintNumberToBaseName.items()} 

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

class ModelParam:
    def __init__(self, s:str, is_variable=False):
        self.s = s
        self.is_variable = is_variable

    def __hash__(self):
        return hash(self.s)

    def __repr__(self):
        return self.s


class ModelTemplate:
    """
    1. Define the model position PyTree
      - Define the keys in the position dict
      - The shapes of the leaf types don't matter because that's defined at run time by the initial position

    Instance Attribute

    position : dict
      scope : str, int, tuple
        param_name

    is_variable : dict
      scope : str, int, tuple
        param_name : bool

    restraints : dict
      scope : str, int, tuple
        param_name : restraint :: (position) -> float
    mover_domain : the default domain of the movers
    """

    def __init__(
        self, 
        mover_domain, 
        position: dict = mutable, 
        proposal: dict = mutable, 
        restraints: dict = mutable,
        restraints_metadata: dict = mutable,
        is_variable: dict = mutable,
        group_ids: list = mutable,
        group_names: list = mutable,
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
        if is_variable is mutable:
            is_variable = {}
        if restraints_metadata is mutable:
            restraints_metadata = {}
        if mover_domain is mutable:
            self.mover_domain = {}

        # TODO: asociate names with groups - a unique group identifier

        self.position = position  # the keys of integer numbers are reserved for node ids
        self.proposal = proposal
        self.restraints = restraints
        self.is_variable = is_variable
        self.mover_domain = mover_domain
        self.group_ids = group_ids
        self.do_checks = do_checks
        self.restraints_metadata = restraints_metadata
        
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
        restraint_key  = _add_restraint(self, scope_key, mapping, logprob_fn, self.do_checks, **options)
        self.add_metadata_to_restraints_metadata(scope_key, restraint_key, 'mapping', mapping)

    def add_dof(self, scope_key, dof_name : str, init_value):
        _assert_fun(dof_name not in self.position[scope_key], f"{dof_name} already in scope", self.do_checks)
        self.position[scope_key][dof_name] = init_value  

    def add_multivariate_normal_restraint(self, 
                                          scope_key, 
                                          x: str,
                                          mean: str,
                                          cov: str,
                                          x_is_variable = False,
                                          mean_is_variable = True,
                                          cov_is_variable = True,
                                          init_positions=mutable,
                                          auto_rename=False,
                                          allow_singular = None,
                                          restraint_base_name=R.wsh.name):
        model_template = self
        do_checks = self.do_checks
        _assert_fun(isinstance(x, str), f'x is not a string', self.do_checks)
        _assert_fun(isinstance(mean, str), f'mean is not a string', self.do_checks)
        _assert_fun(isinstance(cov, str), f'cov is not a string', self.do_checks)
        mapping = {x: 'x', mean : 'mean', cov: 'cov'}
        is_variable = {x: x_is_variable, mean: mean_is_variable, cov: cov_is_variable}

        restraint_key = _add_restraint(model_template=self,
                       scope_key=scope_key,
                       mapping=mapping,
                       logprob_fn=lpdf.multivariate_normal,
                       do_checks=self.do_checks,
                       init_positions=init_positions,
                       restraint_base_name=restraint_base_name,
                       auto_rename=auto_rename,
                       is_variable=is_variable)
        self.add_metadata_to_restraints_metadata(scope_key, restraint_key, 'mapping', mapping)

    def add_wishart_mvn_prior(self,
                              scope_key,
                              cov,  # scatter matrix
                              nu: int, # dof
                              L,  # scale matrix
                              p: int,  # dimensionality
                              cov_is_variable=True,
                              nu_is_variable=False,
                              L_is_variable=False,
                              init_positions=mutable,
                              auto_rename=False,
                              restraint_base_name=R.wsh.name):
        """
        Params :
          scope_key : keys position dict
          cov : covariance matrix
          nu  : dof
          L   : lower cholesky of V scale matrix
          p   : V is (p, p) matrix


        """
        _assert_fun(isinstance(cov, str), f'cov is not a string', self.do_checks)
        _assert_fun(isinstance(nu, str), f'nu is not a string', self.do_checks)
        _assert_fun(isinstance(L, str), f'L is not a string', self.do_checks)

        mapping = {cov:'cov' , nu:'nu' , L:'L' }
        f = partial(ulpdf.wishart_mvn_prior, p=p)
        f.__name__ = "wsh"
        is_variable = {'cov': cov_is_variable, 'nu': nu_is_variable, 'L': L_is_variable}

        restraint_key = _add_restraint(self,
                                       scope_key=scope_key,
                                       mapping=mapping,
                                       logprob_fn=f,
                                       do_checks=self.do_checks,
                                       init_positions=init_positions,
                                       restraint_base_name=restraint_base_name,
                                       auto_rename=auto_rename,
                                       is_variable=is_variable)

    def add_node_group(self, indices, init_params: dict = mutable):
        if init_params is mutable:
            init_params = {}
        _assert_fun(isinstance(init_params, dict), f"init params not a dict", self.do_checks)

        _add_node_group(self, 
                        indices=indices, 
                        init_value=init_params, 
                        do_checks=self.do_checks)
        self.group_ids.append(indices)

    def getgroup(self, groupidx: int):
        return self.group_ids[groupidx]

    def get_pid_dict(self):
        """
        Get a dictionary of key, position ids.
        The key is the scope_key in the position dictionary
        The posiiton id (pid) is the base 0 order of the keys
        """
        pid_dict = {}
        for i, key in enumerate(self.position.keys()):
            pid_dict[key] = i
        return pid_dict

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

    def get_pid(self, scope_key):
        _assert_fun(scope_key in self.position.keys(), f"scope key not in position dict", self.do_checks)
        pid = 0
        for i, key in enumerate(self.position.keys()):
            if key == scope_key:
                break
            pid +=1
        return pid

    def add_metadata_to_restraints_metadata(self, scope_key, restraint_key, meta_key, value):
        if scope_key not in self.restraints_metadata:
            self.restraints_metadata[scope_key] = {}
        if restraint_key not in self.restraints_metadata[scope_key]:
            self.restraints_metadata[scope_key][restraint_key] = {}

        _assert_fun(meta_key not in self.restraints_metadata[scope_key][restraint_key], self.do_checks)
        self.restraints_metadata[scope_key][restraint_key][meta_key] = value

    
class ModFileWriter:
    def __init__(self, model_template):
        self.mt = model_template
        self.rcolumns = columns = ["R", "scope", "name", "form", "*mparams", "*rparams", "*vparams"]

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

    def to_positionid(self):
        pids = ""
        for scope_key in self.mt:
            pids += f"{scope_key}\n"
        return pids

    def get_pid(self, scope_key):
        pid=0
        found = False
        for key in self.mt.position:
            if key == scope_key:
                found = True
                break
            pid +=1

        assert found, f"scope_key not in position dict"
        return pid

    @staticmethod
    def parse_rkey(rkey):
        basename, n = rkey.split("_")
        form = RestraintBaseNameToNumber[basename]
        return basename, form

    @staticmethod
    def rinfo(basename: str):
        form, n_params, rparams = Rinfo[basename].value
        return form, n_params, rparams

    def Rheader(self):
        return "R    scope    name    form    rparams"

    @staticmethod
    def to_rparam_line(rparams: tuple):
        l = ""
        for param in rparams:
            l += f"{param}    "
        return l

    def to_restraint_lines(self):
        lines = ""
        pid_dict = self.mt.get_pid_dict()
        for scope_key, scope_dict in self.mt.restraints.items(): 
            for name, r in scope_dict.items():
                basename, form = self.parse_rkey(name)
                form_, _ , rparams = self.rinfo(basename)
                pid = self.mt.get_pid(scope_key)
                assert form_ == form
                pid = pid_dict[scope_key]
                rparam_line = self.to_rparam_line(rparams)
                lines += f"R    {pid}    {name}    {form}    {rparam_line}\n"
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
    """Return the position array of selected keys from a dictionary d"""
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

def _build_model_score_fn(flist):
    assert len(flist) > 1, f"have more than one restraint before building"
    f0 = flist[0]
    for i in range(1, len(flist)):
        f0 = f0 + flist[i]

    f = f0.f
    s = f"model log density built from {len(flist)} restraints"

    if f.__doc__ is None:
        f.__doc__ = s
    else:
        f.__doc__ += s
    return f


    

def _build_model(model_template, do_checks, position_as_dict):
    if position_as_dict:
        _assert_fun(
            isinstance(model_template.position, dict),
            "position is not a dict",
            do_checks=do_checks,
        )

        flist = []

        for scope_key, rdict in model_template.restraints.items():
            for rkey, restraint in rdict.items():
                flist.append(Function(restraint))

        model_score = _build_model_score_fn(flist)

        return model_template.position, model_score
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

def _add_model_parameter(mt, scope: str, param_name: str, 
                         is_variable = False, 
                         init_position = 1.,
                         do_checks=True):
    """
    Adds a parameter to the model
    """

    if scope not in mt.position:
        mt.position[scope] = {}
    if scope not in mt.is_variable:
        mt.is_variable[scope] = {}

    mt.position[scope][param_name] = init_position
    mt.is_variable[scope][param_name] = is_variable

def _add_restraint_to_restraintsdict(mt, scope, key, logprob_fn, do_checks=True):
    if scope not in mt.restraints:
        mt.restraints[scope] = {}

    _assert_fun(key not in mt.restraints[scope], f"restraint {key} already in scope", do_checks)
    mt.restraints[scope][key] = logprob_fn

def _add_restraints_key(mt, scope, key, logprob_fn, do_checks=True):
    """
    Add a key to the restraints dict. Set the corresponding value to an empty str
    """
    _assert_fun(scope in mt.restraints, f"scope {scope} not in restraints", do_checks)
    _assert_fun(key not in mt.restraints[scope], f"restraint {key} already in scope", do_checks)
    mt.restraints[scope][key] = ""
    

def _add_restraint(
    model_template, 
    scope_key, 
    mapping: dict, 
    logprob_fn, 
    do_checks=True,
    init_positions: dict = mutable,
    restraint_base_name="anon",
    auto_rename= False,
    is_variable: dict = mutable
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
    #print("enter add")
    # 1. If the scope_key in mapping is not in the position dict then the key is added with value init_position 

    if init_positions is mutable:
        init_positions = {}
        for key in mapping:
            init_positions[key] = 1.

    if is_variable is mutable:
        is_variable = {}
        for key in mapping:
            is_variable[key] = False

    for key in mapping:
        _add_model_parameter(model_template, scope_key, key, 
                             is_variable=is_variable[key],
                             init_position= init_positions[key], 
                             do_checks=do_checks) 


#        if key not in model_template.position[scope_key]:
#            model_template.position[scope_key][key] = init_position

    # 2. Build the scoped restraint
    scope_restraint = pyile.build_mapped_fn(mapping, logprob_fn)

    # 3. capture the scope_key by closure
    def restraint(position) -> float:
        return scope_restraint(**position[scope_key])

    docstring = _create_docstring_from_mapping(**locals())
    restraint = _update_function_docstring(restraint, docstring)


    # Update the model template
    #print(model_template, f"model template")
    #model_template.logprob_list.append(restraint)

    updated_dict = model_template.restraints
    if scope_key not in updated_dict:
        updated_dict[scope_key] = {}


    restraint_key = _create_restraint_key(**locals())
    _add_restraint_to_restraintsdict(model_template, scope_key, restraint_key, restraint, do_checks)
    return restraint_key

def _create_restraint_key(restraint_base_name, updated_dict, scope_key, auto_rename: bool = False, **kwargs) -> str:
    fullname = restraint_base_name + "_0"
    if fullname in updated_dict[scope_key]:
        new_name = fullname
        if auto_rename:
            b = 1
            while new_name in updated_dict[scope_key]:
                new_name = f"{restraint_base_name}_{b}"
                b += 1
        else:
            assert False, f"{restraint_base_name} already in scope. Change name or set auto_rename=True"
        return new_name
    else:
        return fullname 
    

def _create_docstring_from_mapping(mapping, logprob_fn, scope_key, **kwargs) -> str:
    """
    mapping
      ModelParamName : ParamValue
    """
    mapped_params_doc = ""
    sig = []
    scope_elements = []
    for scope_element, param_element in mapping.items():
        mapped_params_doc += f"  {scope_element} -> {param_element}\n"
        sig.append(param_element)
        scope_elements.append(scope_element)

    docstring = gen_dynamic_docstring(**locals())
    return docstring

def _update_function_docstring(f, docstring) -> Callable:
    
    if f.__doc__:
        f.__doc__ += docstring
    else:
        f.__doc__ = docstring
    return f


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
