from __future__ import print_function

try:
    import IMP.pynet
    import IMP.pynet.data_in as di
    from IMP.pynet.myitertools import forp, exhaust
except ModuleNotFoundError:
    import pyext.src
    import pyext.src.data_in as di
    from pyext.src.myitertools import forp, exhaust

import IMP.test
import IMP.algebra
import io
import os
import math
from pathlib import Path


class TestDataIn(IMP.test.TestCase):

    synthetic_spec_counts_data = Path(
        "pyext/data/synthetic/41586_2020_2286_MOESM5_ESM.csv"
    )

    def test_read_column_n(self):
        list_len = None
        """ tests data input """
        for column in range(0, 11):
            col_it: Iterator[str] = di.read_column_n(
                self.synthetic_spec_counts_data, column
            )
            col_list = list(col_it)
            print(col_list)
            if list_len:
                self.assertEqual(list_len, len(col_list))
            list_len = len(col_list)

        di.read_column_n(self.synthetic_spec_counts_data, 12)


if __name__ == "__main__":
    IMP.test.main()
