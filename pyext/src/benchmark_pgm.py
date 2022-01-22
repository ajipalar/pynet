import IMP
import jax
import math
from typedefs import (
        Container,
        FilePath,
        PPINetwork, 
        PRNGKey
)


# get_ something that exists
# set_ something that exists
################################################################################
def load_protein_complex(protein_complex: FilePath) -> CartesianTable:
    pass

def get_ppi_network_from_cartesian_table(table: CartesianTable) -> PPINetwork:
    pass

def get_ppi_network_from_complex(protein_complex) -> PPINetwork:
    cartesian_table = load_protein_complex(protein_complex)
    ppi_network = create_ppi_network_from_cartesian_table(table: CartesianTable)
    return ppi_network

def get_synth_apms_data(bait_prey: PPINetwork,
        prng_key: PRNGKey) -> M: 
    """Use the prng_key to generate random APMS
       spectral counts data
       return a bait prey matrix
        r1 r2 r3
       0
       1
       2

       The index is the bait index
       The prey is bait is unknwon
       The calling function is responsible for that

    """
    return y_synth

def poisson_sqr(y, theta, phi) -> float:
    """The Poisson Square Root Graphical Model"""
    return score

def move_poisson_sqr(key: PRNGKey, theta: Vector, phi: Matrix) -> Tuple[Vector, Matrix]:
    theta = jax.random.uniform(key)
    return theta, phi

def eta_1(s, i):
    pass 

def eta_2(s, i):
    pass

def base_measure(x, s, i):
    pass

def Anode(eta1, eta2, s, i):
    pass

def Lambda():
    pass

def magPhi():
    pass


def A(theta, phi):
    """As Inoyue section 3 equation 4
    https://www.davidinouye.com/publication/inouye-2016-square/inouye-2016-square.pdf
    """

    def B(x):
        fac = math.factorial(x)
        return - jnp.log(fac)

    def exponant(theta: Vector, phi: Matrix, x: Vector) -> float:
        sqrtx = jnp.sqrt(x)
        t1 = theta @ sqrtx
        t2 = sqrtx @ phi @ sqrtx
        t4 = sum(B(x(s)) for s in range(S))
        s = t1 + t2 + t3 + t4
        return jnp.exp(s)

    # Compute a "slice sample" for node conditional distributions
    # The bounds are computed in closed form

    #Use the slice sample to develop a Gibbs sampler

    #Derive an annealed importance sampler

    #Use the Gibbs as an intermediate sampler

    #Combine off-diagonal part phi_off

    #Linearly changing gamma from 0 to 1

    #Start from base exponential

    #Pr(x|eta1=phi_ss, eta_2=0) -> Pr(x|theta, phi)
        
        
        




   




"""
Glossary:
    This is a declartion
    The program has a Model
    A Model has Hierarchy(s)
    A Model has a ParticleIndex
    A Model has Particles
    A Model has Decorators
    A Model has a ScoringFunction
    A Model has Singletons
    A Model has Containers
    Decorators have Particles
    Particles have Attributes
    A Hierarchy has child Particles
    A Singleton has a Particle
    A Container has a group of related Particles
    A Modifier changes the values of the attirbutes of a one or more Particles
    A Predicate operates on Particles
    A Restraint operates on Particles
    Restraints have Scores
    A Constraint is invariant
    A ScoreState is a Constraint
    A Score operates on ones to four Particles
    A Mover has Particles
    Some children have Leaves

"""

