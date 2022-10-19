from _core import  *
import numpy as np
import jax.numpy as jnp
import lpdf
from inspect import signature
from contextlib import contextmanager

n_examples = 2
def example1():
    group = (0, 1, 2) 
    A = {'a': 0, 'b': 1}
    B = {'c': 2, 'd': 3}
    C = A | B
    
    context_to_signature = {'a':'rho', 'b':'sigma'}
    
    
    init_position = {0:{}, 1:{}, 2:{}} 
    
    def logprob_example(rho, sigma, /, **kwargs):
        return jnp.log(rho) + jnp.log(sigma)
    
    
    
    model_template = ModelTemplate(init_position)
    point_name = 9
    add_point(model_template, point_name)
    #add_node_index(model_template, 0)
    
    model_template.add_node_group(group)
    #model_template.build()
    
    model_template.add_restraint(group, context_to_signature, logprob_example)
    return model_template

def example2():

    mt = ModelTemplate()
    print(f"{mt} example 2")
    mt.add_contiguous_nodes(0, 100)
    #print(mt.position)
    try:
        mt.add_node_group((200, 300, 400), {})
        raise NameError
    except AssertionError:
        print("caught")

    mt.add_point(200)
    mt.add_point(300)
    mt.add_point(400)
    mt.add_node_group((200, 300, 400), {})
    print('building...')
    mt.build()
    
    print('built')
    mod_writer = ModFileWriter(mt)
    print(mod_writer.to_modfile())
    print(mt.restraints)
     
def example3():
    print('Example 3')
    m = ModelTemplate()
    r = lpdf.beta
    mapping = {'v_alpha': 'a', 'v_beta':'b', 'v_prob':'x', 'c_betaloc':'loc', 'c_betascale':'scale'}
    scope_key = 0
    m.add_point(scope_key)
    m.add_restraint(scope_key, mapping, r)
    return m, mapping

def example4():
    print("example4")
    m = ModelTemplate()
    mapping = {'mu':'loc', 'sigma':'scale', 'x':'x'}
    m.add_point(100)
    m.add_restraint(100, mapping, lpdf.norm)


    return m, mapping

def example5():
    mapping = {'v_alpha': 'a', 'v_beta':'b', 'v_prob':'x', 'c_betaloc':'loc', 'c_betascale':'scale'}
    m = ModelTemplate()
    scope_key = 0
    m.add_point(scope_key)
    return m, mapping, scope_key

def example6():
    """
    ModelParam Class
    """

    p = ModelParam('p', True)
    d = {p: 0}
    return d

def example7():
    m = ModelTemplate()
    m.add_contiguous_nodes(0, 100)
    m.add_node_group(tuple(range(0, 100)))
    m.add_multivariate_normal_restraint(m.group_ids[0], x="y", mean="muy", cov="covy")
    #m.add_multivariate_normal_restraint(m.group_ids[0], x="y", mean="muy", cov="covy")  # this throws an error to user as it should
    m.add_multivariate_normal_restraint(m.group_ids[0], x="y", mean="muy", cov="covy", auto_rename=True)

    return m

def example8():
    m = example7()
    w = ModFileWriter(m)
    s = w.to_restraint_lines()
    h = w.Rheader()

    return m, w, s, h


def run_examples():
    start=1
    stop = 8
    for i in range(start, stop):
        exec(f"example{i}()")
    

