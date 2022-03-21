import IMP.algebra
from pyext.src.myitertools import exhaust as exhaust, forp as forp
from typing import Any

class TestDataIn(IMP.test.TestCase):
    synthetic_spec_counts_data: Any
    def test_read_column_n(self) -> None: ...
