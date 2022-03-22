from test.src import two_namespace_example as td, NullModule
from typing import Any
import IMP.test

Module = Any
class BaseExample(IMP.test.TestCase):
    m: Module = NullModule
    def test_example_testdef(self): 
        x = 2.
        td.example_testdef(x, self.m)


