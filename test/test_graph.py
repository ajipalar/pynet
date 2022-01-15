from __future__ import print_function
import IMP
import IMP.test
import IMP.algebra
import unittest
import sys

try:
    import IMP.pynet.graph as graph
    from IMP.pynet.graph import Node 
except:
    import pyext.src.graph as graph
    from pyext.src.graph import Node

class TestGraph(IMP.test.TestCase):
    def test_print(self):
        pass
    def test_nnodes(self):
        self.assertEqual(Node.nnodes , 0)
        x = Node()
        self.assertEqual(Node.nnodes , 1)
        y = Node(structure=0)
        self.assertEqual(Node.nnodes , 2)

    @IMP.test.expectedFailure
    def test_fail(self):
        self.assertEqual(0, 1)

    @unittest.expectedFailure
    def test_fail2(self):
        self.assertEqual(0, 1)

    def test_fail3(self):
        self.assertEqual(0, 1)

class TestEID(IMP.test.TestCase):
    @IMP.test.expectedFailure
    def test_fail(self):

#These variables are provided for interactive testing w/ the python interpreter
#Use test_graph.run_tests(test_list)

        self.assertEqual(0, 1)

    @unittest.expectedFailure
    def test_fail2(self):
        self.assertEqual(0, 1)

    def test_fail3(self):
        self.assertEqual(0, 1)

if __name__ == "__main__":
    IMP.test.main()
