from __future__ import print_function

from .casedefs.test_ais import(
    DevPropertyTrivial,
    DevTrivialAIS,
    DevTrivialBetaDependantAIS,
    DevTestAIS
)

import IMP.pynet

#import the tests

#run tests
if __name__ == '__main__':
    IMP.test.main()
