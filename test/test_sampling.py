PRINT_ON = False
printf = lambda s: print(s) if PRINT_ON else None
printf("Importing modules...")
import sys
#sys.path.append("../examples")
import numpy as np
#printf("Done")
if __name__ == "__main__":
    import IMP.pynet.sampling as samp
else:
    import pyext.src.sampling as samp

class TestSampling(IMP.test.TestCase):

    def test_degree_prior(self):
        """ Testing degree prior. """
        a = np.arange(10)
        b = np.arange(10)
        self.assertEqual(samp.degree_prior(a, b), 0)
        self.assertEqual(samp.degree_prior(a, b, s=10), 0)
        self.assertEqual(samp.degree_prior(a, b, s=-1), 0)
        self.assertEqual(samp.degree_prior(a, b, s=-1), 0)
        self.assertEqual(samp.degree_prior(a, b, s=-1), 0)
        self.assertEqual(samp.degree_prior(a, b, s=-1), 0)
        b = b + 1

        self.assertEqual(samp.degree_prior(a, b), 10)
        self.assertEqual(samp.degree_prior(a, b, s=0.5), 40)
        self.assertEqual(samp.degree_prior(a, b, s=100), ((1 / 100)**2)*10)

        print("Passed")
    
    def test_e_base(self):
        """e_base should return the base edge id. """
        for v in range(2, 13):
            for i in range(v):
                if i < v-1:
                    self.assertEqual(samp.e_base(i, v), True)
    
    def test_edge_id(self):
        """edge_id should return the unique edge identifier"""
        for v in range(2, 101):
            printf(f"Testing size {v} graph")
            Emax = v*(v-1)//2
            entered_outer_loop = False
            entered_inner_loop= False

            for i in range(v):
                entered_outer_loop = True
                for j in range(i+1, v):
                    entered_inner_loop = True
                    eid = samp.get_immutable_edge_id(i, j, v)
                    if v==2:
                        printf(f"{i, j, eid}") 
                    s2, t2 = samp.edge_from_eid(eid, v)

                    self.assertLessEqual(0, i)
                    self.assertEqual(i, s2)
                    self.assertLess(s2, j)
                    self.assertEqual(j, t2)
                    self.assertLess(eid, Emax)
                    if (i == v  - 1) and (t==v):
                        self.assertEqual(eid, Emax)
            self.assertTrue(entered_outer_loop)
            self.assertTrue(entered_inner_loop)
    
    def test_plot_dataset_overlap(self):
        a = {'a', 'b', 'c'}
        b = {'a', 'b', 'c'}
        c = {'a', 1}
        d = {}
        e = {1, 2, 3}
        f = {6, 7, 8}
        g = {'A', 'B', 'C'}
        h = {}
        self.assertEqual(False)
    
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
        
    @expectedFailure
    def test_fail2(self):
        self.assertEqual(1, 2)

    def test_fail3(self):
        self.assertEqual(1, 2)

if __name__ == "__main__":
    IMP.test.main()
