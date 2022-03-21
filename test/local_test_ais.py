from __future__ import print_function
import IMP.test


#Define the import paths
import pyext.src.ais as ais
import pyext.src.functional_gibbslib as fg
import pyext.src.PlotBioGridStatsLib as bsl
import pyext.src.distributions as dist

#import the tests
from ._test_ais import (
    DevPropertyTrivial,
    DevTrivialAIS,
    DevTrivialBetaDependantAIS,
    DevTestAIS
)

#run the tests
if __name__ == '__main__':
    IMP.test.main()
