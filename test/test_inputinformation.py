print(f'Name: {__name__}')
print(f"Package: {__package__}")
import sys
print(sys.path)

import numpy as np
from ..net.inputinformation import get_dataset_overlap

def test_get_dataset_overlap():
    a = {'a', 'b', 'c'}
    b = {'a', 'b', 'c'}
    c = {'a', 1}
    d = {}
    e = {1, 2, 3}
    f = {6, 7, 8}
    g = {'A', 'B', 'C'}
    h = {}
    l = [a, b, c, d, e, f, g, h]
    k = 'abcdefgh'

    dataset_dict = dict(zip(k, l ))
    m = ii.get_dataset_overlap(dataset_dict, dataset_names=dataset_dict.keys())

    #Check that the diagonal elements of the matrix are equal to the length of the ipput
    for i in range(len(m)):
        assert m[i, i] == len(l[i])

    #Check for symmetry
    assert np.equal(m, m.T).all()

    #Test A
    a_inter = [3, 3, 1, 0, 0, 0, 0, 0]
    for i in range(len(m)):
        assert m[i, 0] == a_inter[i]
        assert m[0, i] == a_inter[i]

def run_tests():
    test_get_dataset_overlap()

if __name__ == "main":
    run_tests()


