# Import modules
from __future__ import print_function
import numpy as np
import jax
import jax.numpy as jnp

# Load synthetic datasynthetic data
synthetic_spec_counts_data = np.random.randint(0, 200, (300, 3))

print(synthetic_spec_counts_data)
# Set up model
def represent_network(n):
    """represent a network of n nodes."""
    return network


"""
Vertex sets have restraints
Restraints have vertex sets
Vertex sets have data

Restraint 1 applies to vertex set 1
Restraint 2 applies to vertex set 2
Restraint n applies to vertex set n

None of the restraints need to know about any of the other restraints
"""


def register_restraints():
    """
    Score Table:
    ID | functor | *args (vertex_set, data, PRNGKey, other_params)
    """


def poisson_sqrgm(vertex_set, data):
    return score


def sample(M, D, PRNGKey):
    return samples


def plot_MCMC_chain(samples):
    """x: step, y: score"""
    return None


def check_R_hat(samples):
    return R_hat


def KL_divergence_test(samples):
    return KL_statistic


def plot_sample_edge_mean(samples):
    return None


def plot_sample_edge_variance(samples):
    return None


def cluster_models(samples):
    return clustered_samples
