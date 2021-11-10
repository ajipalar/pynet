"""
->   Define a graph prior
->   Define a graph likelihood
->   Define a forward model
->   Define an uncertainty model
->   Define the inverse likelihood function
->   

"""
import graph_tool as gt
from dev import *

apms = pd.read_excel(apms_gordon)


#Represent the model


#Define the prior


#Define the likelihood
def l_spec_counts_apms_forward_model(g:G, theta) -> d_i: 
    """
    Likelihood for triplets
    g: a triplet X, Y, Z
    """





    


#Define the scoring function


#Define a graph enumerator


#Create synthetic data from the inverse of the likelihood function


#Prior predictive check on synthetic data




#Enumerate all graphs and see how they score





