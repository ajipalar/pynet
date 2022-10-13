from _core import  *
import numpy as np
import jax.numpy as jnp
from inspect import signature

group = (0, 1, 2) 
A = {'a': 0, 'b': 1}
B = {'c': 2, 'd': 3}
C = A | B

context_to_signature = {'a':'rho', 'b':'sigma'}


init_position = {0:{}, 1:{}, 2:{}} 

def logprob_example(rho, sigma, /, **kwargs):
    return jnp.log(rho) + jnp.log(sigma)



model_template = ModelTemplate(init_position)
point_name = 'a'
add_point(model_template, point_name)
#add_node_index(model_template, 0)

add_node_group(model_template, group)
model_template.build()

model_template.add_restraint(group, context_to_signature, logprob_example)
model_template.help_restraint(0)
# Clean up example

# Next example



