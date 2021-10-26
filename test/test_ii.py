import config
import utilities.utils as utils
from .testoptions import decorate_print_test_results

if config.PRINT_MODULE_INFO:
    utils.moduleinfo(locals())
   

if __name__ != "__main__":
    assert config.RUN_ALL_NET_TESTS == True
    from net.ii import get_dataset_overlap
import sys
import numpy as np


def test_get_dataset_overlap():
    a = {'a', 'b', 'c'}
    b = {'a', 'b', 'c'}
    c = {'a', 1}
    d = set()
    e = {1, 2, 3}
    f = {6, 7, 8}
    g = {'A', 'B', 'C'}
    h = set()
    l = [a, b, c, d, e, f, g, h]
    k = 'abcdefgh'

    dataset_dict = dict(zip(k, l ))
    
    m = get_dataset_overlap(dataset_dict, dataset_names=dataset_dict.keys())

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
    test_list = [test_get_dataset_overlap]
    for test in test_list:
        if config.PRINT_TEST_RESULTS:
           test = decorate_print_test_results(test, __name__)
        test()

if __name__ == "main":
    run_tests()


