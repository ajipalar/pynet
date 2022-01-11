from __future__ import print_function
from pathlib import Path
import unittest
import sys
from pynet_test_config import impure_pynet_testing_path_setup

impure_pynet_testing_path_setup()

from IMP.pynet.graph import Node 

class TestGraph(unittest.TestCase):
    def test_print(self):
        pass
    def test_nnodes(self):
        self.assertEqual(Node.nnodes , 0)
        x = Node()
        self.assertEqual(Node.nnodes , 1)
        y = Node(structure=0)
        self.assertEqual(Node.nnodes , 2)

class TestEID(unittest.TestCase):
    pass

if __name__ == "__main__":
    unittest.main()
