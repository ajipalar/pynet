#PRINT_ON = False
#printf = lambda s: print(s) if PRINT_ON else None
#printf("Importing modules...")
import sys
#sys.path.append("../examples")
import IMP.pynet.sampling as s
import numpy as np
#printf("Done")

class TestSampling(IMP.test.TestCase):

    def test_magnitude(self):
        """Write the test cast, print statemetns ok"""
        pass
    def test_degree_prior(self):
        """
        Testing degree prior
        """
        a = np.arange(10)
        b = np.arange(10)
        self.assertEqual(s.degree_prior(a, b), 0)
        self.assertEqual(s.degree_prior(a, b, s=10), 0)
        self.assertEqual(s.degree_prior(a, b, s=-1), 0)
        self.assertEqual(s.degree_prior(a, b, s=-1), 0)
        self.assertEqual(s.degree_prior(a, b, s=-1), 0)
        self.assertEqual(s.degree_prior(a, b, s=-1), 0)
        b = b + 1

        self.assertEqual(s.degree_prior(a, b), 10)
        self.assertEqual(s.degree_prior(a, b, s=0.5), 40)
        self.assertEqual(s.degree_prior(a, b, s=100), ((1 / 100)**2)*10)
)
        print("Passed")
    
    def test_e_base(self):
        """
        Testing e_base
        """
        for v in range(2, 13):
            for i in range(v):
                if i < v-1:
                    self.assertEqual(s.e_base(i, v), True)
    
    def test_edge_id(self):
        """Testing edge id"""
        for v in range(2, 101):
            printf(f"Testing size {v} graph")
            Emax = v*(v-1)//2
            for i in range(v):
                for j in range(i+1, v):
                    eid = s.get_immutable_edge_id(i, j, v)
                    if v==2:
                        printf(f"{i, j, eid}") 
                    s2, t2 = s.edge_from_eid(eid, v)
                    self.assertLessEqual
                    assert 0 <= i == s2 < j == t2
                    assert eid < Emax
                    if (i == v  - 1) and (t==v):
                        assert eid == Emax
    
    def test_plot_dataset_overlap(self):
        a = {'a', 'b', 'c'}
        b = {'a', 'b', 'c'}
        c = {'a', 1}
        d = {}
        e = {1, 2, 3}
        f = {6, 7, 8}
        g = {'A', 'B', 'C'}
        h = {}
    
    def run_tests(self):
        """
        Running tests
        
        """
        tests = [test_degree_prior,
                 test_e_base,
                 test_edge_id,
                 ]
        for test in tests:
            printf(test.__doc__)
            test()

    @unittest.expectedFailure
    def test_fail(self):
        self.assertEqual(1, 2)

if __name__ == "__main__":
    unittest.main()
