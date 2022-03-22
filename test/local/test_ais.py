from __future__ import print_function
import IMP.test

import pyext.src.ais as ais
import pyext.src.distributions as dist

from ..src.casedef_ais import(
    DevPropertyTrivial,
    DevTrivialAIS,
    DevTrivialBetaDependantAIS,
    DevTestAIS
)

class l1(DevPropertyTrivial):
    dist = dist
    ais = ais

class l2(DevTrivialAIS):
    dist = dist
    ais = ais

class l3(DevTrivialBetaDependantAIS):
    dist = dist
    ais = ais

class l4(DevTestAIS):
    dist = dist
    ais = ais



#Define the import paths

#import the tests

#run the tests
if __name__ == '__main__':
    IMP.test.main()
