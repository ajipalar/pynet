from ..testdefs import two_namespace_example as td
from . import NullModule
from typing import Any
import IMP
import IMP.test

Module = Any
class BaseExample(IMP.test.TestCase):
    m: Module = NullModule
    def test_example_testdef(self): 
        x = 2.
        td.example_testdef(x, self.m)


