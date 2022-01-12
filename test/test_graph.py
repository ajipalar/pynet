from __future__ import print_function
import IMP.test
import IMP.algebra
import unittest
import sys

from IMP.pynet.graph import Node 

class TestGraph(IMP.test.TestCase):
    def test_print(self):
        pass
    def test_nnodes(self):
        self.assertEqual(Node.nnodes , 0)
        x = Node()
        self.assertEqual(Node.nnodes , 1)
        y = Node(structure=0)
        self.assertEqual(Node.nnodes , 2)

    @expectedFailure
    def test_fail(self):
        self.assertEqual(0, 1)

    @unittest.expectedFailure
    def test_fail2(self):
        self.assertEqual(0, 1)

    def test_fail3(self):
        self.assertEqual(0, 1)

class TestEID(IMP.test.TestCase):
    @expectedFailure
    def test_fail(self):
        self.assertEqual(0, 1)

    @unittest.expectedFailure
    def test_fail2(self):
        self.assertEqual(0, 1)

    def test_fail3(self):
        self.assertEqual(0, 1)

if __name__ == "__main__":
    IMP.test.main()
