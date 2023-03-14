# PyNet: Probabilistic Inference over Networks

## What is PyNet?

PyNet is Bayesian inference over undirected graphs using [JAX](https://github.com/google/jax).
PyNet is a stand-alone python module and part of the [Integrative Modeling Platform](https://integrativemodeling.org/) ([IMP](https://github.com/salilab/imp)).

A fork of PyNet is [here](https://github.com/salilab/pynet)

## Modeling in PyNet

A PyNet model consists of four-stages
  1. Defining the inputs
  2. Defining the number of nodes in the network, nuisance parameters, and a log scoring function 
  3. Sampling alternative configurations of the edges and nuisance parameters to produce an ensemble of network models
  4. An analysis of the ensemble 

## Status

PyNet is under activate development and is currently pre-release 2023/03/13

_Author(s)_: (Aji Palar)

_Maintainer_: (Ben Webb)

_License_: LGPL 2.1

_Publications_:
- None



