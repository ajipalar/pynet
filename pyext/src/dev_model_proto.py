import numpy as np
import model_proto as mp
import jax
import jax.numpy as jnp
from functools import partial

n = 10
A = np.zeros((n, n), dtype=np.int32)
A[(1, 2, 3, 4, 5), (0, 0, 0, 1, 2)] = 1
A = np.tril(A, k=-1)
A = A + A.T
Cs = np.array([0, 1, 3], dtype=np.int32)
A = jnp.array(A)
len_A = len(A)
assert len_A == n

Ss = np.array([1, 0.7, 0.6, 0.8, 0.5, 0.3, 0.2, 0.1, 0.4, 0.05])
assert len(Ss) == n
key = jax.random.PRNGKey(13)
pe = mp.get_possible_edges(n)

def test_flip():
    flip = mp.flip

    jflip = jax.jit(flip)

    assert jflip(0) == flip(0)
    assert flip(1) == jflip(1)
    assert jflip(1) != flip(0)
    assert flip(0) != flip(1)
    assert jflip(0) != jflip(1)


def test_flip_with_prob():
    def test_vf(vf): 
        assert jnp.all(vf(keys, z, z) == z)
        assert jnp.all(vf(keys, z, o) == o)
        assert jnp.all(vf(keys, o, o) == z)
        assert jnp.all(vf(keys, o, z) == o)

    f = mp.flip_with_prob
    jf = jax.jit(mp.flip_with_prob)

    assert f(key, 0, 0) == 0
    assert f(key, 0, 1) == 1
    assert f(key, 1, 0) == 1
    assert f(key, 1, 1) == 0

    assert jf(key, 0, 0) == 0
    assert jf(key, 0, 1) == 1
    assert jf(key, 1, 0) == 1
    assert jf(key, 1, 1) == 0

    vf = jax.vmap(f)

    l = 9
    z = jnp.zeros(l, dtype=jnp.int32)
    o = jnp.ones(l, dtype=jnp.int32)
    keys = jax.random.split(key, l)



    jvf = jax.jit(vf)
    test_vf(vf)
    test_vf(jvf)

def test_flip_edges():
    l = 9
    m = l * (l - 1) // 2
    z = jnp.zeros(l, dtype=jnp.int32)
    o = jnp.ones(l, dtype=jnp.int32)

    f = mp.flip_edges
    f = partial(f, len_edge_vector=l)
    jf = jax.jit(f)

    z_a = f(key, z, z)
    z_b = jf(key, z, z)
    z_c = f(key, o, o)
    z_d = jf(key, o, o)

    assert jnp.all(z_a == z_b)
    assert jnp.all(z_b == z_c)
    assert jnp.all(z_c == z_d)
    assert jnp.all(z_d == z)

    o_a = f(key, z, o)
    o_b = jf(key, z, o)
    o_c = jf(key, o, z)
    o_d = f(key, o, z)

    assert jnp.all(o_a == o_b)
    assert jnp.all(o_b == o_c)
    assert jnp.all(o_c == o_d)
    assert jnp.all(o_d == o)

def test__select_n_random_edges(
        n_edges = 10,
        len_A = 10):

    f = mp._select_n_random_edges
    pe = mp.get_possible_edges(len_A)
    f = partial(f, n_edges=n_edges, len_A=len_A)
    jf = jax.jit(f)

    edges = f(key, pe)
    jedges = jf(key, pe)

    assert jnp.all(edges == jedges), (edges, jedges)
    if n_edges > 0:
        assert edges.shape == (n_edges, 2), edges.shape
    else:
        assert edges.shape == (0,)
    assert jnp.all(edges < len_A)
    assert jnp.all(jedges < len_A)
    assert jnp.all(edges > -1)
    assert jnp.all(jedges > -1)

def test_flip_adjacency__j():
    f = mp.flip_adjacency__j
    f = partial(f, possible_edges=pe,n_edges=10, len_A=len(A))

    jf = jax.jit(f)

    a = f(key, A, 0.5)
    b = jf(key, A, 0.5)

    c = f(key, A, 0)
    d = jf(key, A, 0)
    e = f(key, A, 0.0)
    g = f(key, A, 0.0)

    assert jnp.all(a == b)
    assert jnp.all(c == d)
    assert jnp.all(c == A)
    assert jnp.all(c == e)
    assert jnp.all(g == e)

    h = f(key, A, 1.)
    i = jf(key, A, 1.)
    j = f(key, A, 1)
    k = jf(key, A, 1)

    assert jnp.all(h == i)
    assert jnp.all(i == j)
    assert jnp.all(j == k)

    # flip all edges

    f = mp.flip_adjacency__j
    f = partial(f, possible_edges=pe, n_edges=len_A * (len_A - 1) // 2, len_A=len_A)

    jf = jax.jit(f)

    a = f(key, A, prob=1.)
    b = jf(key, A, prob=1.)
    
    assert jnp.all(a == b)

    tril_indices = jnp.tril_indices(len_A, k=-1)

    La = a[tril_indices]
    LA = A[tril_indices]

    # flipped every edge
    assert jnp.all(La != LA), (La, LA)





    



    

def do_tests():
    test_flip()
    test_flip_with_prob()
    test_flip_edges()

    test__select_n_random_edges(0, 10)
    test__select_n_random_edges(1, 10)
    test__select_n_random_edges(0, 9)
    test__select_n_random_edges(1, 9)
    expected_failure = False
    
    try:
        test__select_n_random_edges(7, 3)
    except ValueError:
        expected_failure = True

    assert expected_failure

    test_flip_adjacency__j()








