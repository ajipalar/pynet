import unittest
import sys
sys.path.append("..")

from net.graph import Node

class TestGraph(unittest.TestCase):
    def test_print(self):
        print("Unit tests running")
    def test_nnodes(self):
        print("testing nnodes")
        assert Node.nnodes == 0
        x = Node()
        assert Node.nnodes == 1
        y = Node(structure=0)
        assert Node.nnodes == 2

if __name__ == "__main__":
    unittest.main()