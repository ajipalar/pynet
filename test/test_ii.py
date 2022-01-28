import sys
import numpy as np
import IMP
import unittest

#Module specific imports
try:
    import IMP.test
    import IMP.algebra
    import IMP.pynet.config
    import IMP.pynet.utils as utils
    from IMP.pynet.ii import get_dataset_overlap
except ModuleNotFoundError:
    import pyext.src.config
    import pyext.src.utils as utils
    from pyext.src.ii import get_dataset_overlap


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

    def test_neg(self):
        self.assertEqual(False, True)

    def test_neg2(self):
        self.assertNotEqual(False, True)

    @unittest.skip
    def test_fail3(self):
        self.assertEqual(False, True)

    def test_pass(self):
        self.assertEqual(True, True)


if __name__ == "main":
    IMP.test.main()
