from __future__ import print_function
import argparse
from pathlib import Path
import unittest
import sys

from IMP.pynet.graph import Node 

#impure_pynet_testing_path_setup()

#Testing setup
"""
parser = argparse.ArgumentParser(description='Local testing toggle.')
parser.add_argument('-local-test', '--local_test') 
args = parser.parse_args()
"""

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
