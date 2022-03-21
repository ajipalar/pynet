from __future__ import print_function
import IMP.test
import IMP.algebra

import IMP.pynet
import IMP.pynet.ais as ais
import IMP.pynet.functional_gibbslib as fg
import IMP.pynet.PlotBioGridStatsLib as bsl
import IMP.pynet.distributions as dist

#import the tests
from ._test_ais import (
    DevPropertyTrivial,
    DevTrivialAIS,
    DevTrivialBetaDependantAIS,
    DevTestAIS
)

#run tests
if __name__ == '__main__':
    IMP.test.main()
