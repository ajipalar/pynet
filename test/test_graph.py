from __future__ import print_function

try:
    import IMP.pynet.graph as graph
    from IMP.pynet.graph import Node
except ModuleNotFoundError:
    import pyext.src.graph as graph
    from pyext.src.graph import Node
import IMP
import IMP.test
import IMP.algebra
import unittest
import sys


class TestGraph(IMP.test.TestCase):
    module = graph

    def test_print(self):
        pass

    def test_nnodes(self):
        self.assertEqual(Node.nnodes, 0)
        x = Node()
        self.assertEqual(Node.nnodes, 1)
        y = Node(structure=0)
        self.assertEqual(Node.nnodes, 2)

    def test_equal(self):
        self.assertEqual(1, 1)

    def test_neg(self):
        self.assertNotEqual(0, 1)

    def test_class_names(self):
        exceptions = []
        words = []
        self.assertClassNames(self.module, exceptions, words)

    def test_function_names(self):
        exceptions = []
        words = []
        self.assertFunctionNames(self.module, exceptions, words)


if __name__ == "__main__":
    IMP.test.main()
