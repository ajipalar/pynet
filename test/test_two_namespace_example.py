import IMP
import IMP.test
from ..casedefs.test_two_namespace_example import BaseExample

import pyext.src.distributions as dist


class TestExample(BaseExample):
    m = dist


if __name__ == "__main__":
    IMP.test.main()
