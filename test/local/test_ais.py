from __future__ import print_function
import IMP.test

import pyext.src.ais as ais
import pyext.src.distributions as dist

from ..src import casedef_ais as test_case


class T1(test_case.UnitTest1):
    dist = dist
    ais = ais


"""
import(
    DevPropertyTrivial,
    DevTrivialAIS,
    DevTrivialBetaDependantAIS,
    DevTestAIS
)
class l1(test_case.DevPropertyTrivial):
    dist = dist
    ais = ais

class l2(test_case.DevTrivialAIS):
    dist = dist
    ais = ais
"""


class l3(test_case.DevTrivialBetaDependantAIS):
    dist = dist
    ais = ais


class l4(test_case.DevTestAIS):
    dist = dist
    ais = ais


# Define the import paths

# import the tests

# run the tests
if __name__ == "__main__":
    IMP.test.main()
