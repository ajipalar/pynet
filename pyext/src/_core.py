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

Attribute = namedtuple("Attribute", "name val")


    


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



    def __init__(self, name : str, idxs : NodeIndices, model : Model):
        self.name = name
        self.idxs = idxs 



def _check_restraint_name(restraint : Restraint, model : Model):
    assert restraint.name not in model.restraints

def add_restraint(restraint : Restraint, model : Model, do_checks=True):

    _check_restraint_name(restraint, model) if do_checks else None





class RestraintAdder:
    """
    Adds a restraint to a model

    Does some checking
    """

    def __init__(self, restraint : Restraint, model : Model):
        self.restraint = restraint
        self.model = model

    def check_name_is_unique(self):
        assert self.model





        

class NodeIndex(int): ...

class NodeIndices(npt.ArrayLike): ...

class Node:
    """
    Attributes should have a unique hashable name.
    """

    idx : NodeIndex
    attributes : namedtuple # An unordered collection
    group_attributes : namedtuple # A collection of group attributes


    

    def __init__(self, idx : NodeIndex, attributes : Attributes):
        ...

class Nodes(dict): ...
class RestraintName(str): ...

ModelState = namedtuple("ModelState", "consts variables scores")

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

    def __init__(self, nodes : Nodes, restraints: dict[RestraintName, Restraint] = {}): 
        self.restraints = restraints  #restraint.name : restraint

    node_attributes : dict[NodeIndex : Attribute] 
    node_group_attributes : dict[OrderedNodeSet : Attribute]
    nodes : dict[NodeIndex : Node]

    def add_attribute_to_node(self, idx: NodeIndex, attribute : Attribute):

        node_attributes : dict = self.nodes[idx].attributes._asdict()
        assert attribute.name not in node_attributes, f"{attribute.name} already in node {idx}"
        node_attributes = node_attributes | attribute._as_dict() 
        assert attribute.name in node_attributes, f"{attribute.name} should be in node {idx}"



    def add_n_nodes(n):
        self.node_index = np.arange(0, n)

    def add_node_properties(node_index, prop):
        ...

    def add_data(node_indices):
        ...


    def add_restraint():
        ...


