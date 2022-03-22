import IMP.test
from test.src.casedef_two_namespace_example import(
    BaseExample
)

import pyext.src.distributions as dist

class TestExample(BaseExample):
    m = dist

if __name__ == "__main__":
    IMP.test.main()

