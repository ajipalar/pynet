PRINT_ON = False
printf = lambda s: print(s) if PRINT_ON else None
printf("Importing modules...")
import sys
sys.path.append("../examples")
import IMP.pynet.sampling as s
import numpy as np
printf("Done")

def test_degree_prior():
    """
    Testing degree prior
    """
    a = np.arange(10)
    b = np.arange(10)
    assert s.degree_prior(a, b) == 0
    assert s.degree_prior(a, b, s=10) == 0
    assert s.degree_prior(a, b, s=-1) == 0
    b = b + 1
    assert s.degree_prior(a, b) == 10
    assert s.degree_prior(a, b, s=0.5) == 40
    assert s.degree_prior(a, b, s=100) == ((1 / 100)**2)*10
    printf("Passed")

def test_e_base():
    """
    Testing e_base
    """
    for v in range(2, 13):
        for i in range(v):
            if i < v-1:
                try:
                    assert s.e_base(i, v) 
                except AssertionError:
                    printf(f"{i, v, s.e_base(i, v)}")

def test_edge_id():
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
                assert 0 <= i == s2 < j == t2
                assert eid < Emax
                if (i == v  - 1) and (t==v):
                    assert eid == Emax

def test_plot_dataset_overlap():
    a = {'a', 'b', 'c'}
    b = {'a', 'b', 'c'}
    c = {'a', 1}
    d = {}
    e = {1, 2, 3}
    f = {6, 7, 8}
    g = {'A', 'B', 'C'}
    h = {}

def run_tests():
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

if __name__ == "__main__":
     run_tests()
