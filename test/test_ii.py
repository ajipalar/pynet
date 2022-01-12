#from pynet_test_config import impure_pynet_testing_path_setup

#impure_pynet_testing_path_setup()
import sys
import numpy as np

#Module specific imports
import IMP.pynet.config
import IMP.pynet.utils as utils
from IMP.pynet.ii import get_dataset_overlap
#from testoptions import decorate_print_test_results

#if config.PRINT_MODULE_INFO:
#    utils.moduleinfo(locals())
   



class TestII(IMP.test.TestCase):

    def test_get_dataset_overlap(self):
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
            self.assertEqual(m[i, i] == len(l[i]))
    
        #Check for symmetry
        result_bool = np.equal(m, m.T).all()
        self.assertEqual(True, result_bool)
    
        #Test A
        a_inter = [3, 3, 1, 0, 0, 0, 0, 0]
        for i in range(len(m)):
            self.assertEqual(m[i, 0], a_inter[i])
            self.assertEqual(m[0, i], a_inter[i])
    
    def test_fail(self):
        self.assertEqual(False, True)
    def test_pass(self):
        self.assertEqual(True, True)

    def run_tests():
        test_list = [test_get_dataset_overlap]
        for test in test_list:
            if config.PRINT_TEST_RESULTS:
               test = decorate_print_test_results(test, __name__)
            test()

if __name__ == "main":
    unittest.main()


