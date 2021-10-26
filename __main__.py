import sys
import argparse
import config

parser = argparse.ArgumentParser(description='Debug the runscript')
parser.add_argument('-t', '-test', help='run all tests', action='store_const', const=True)
parser.add_argument('-m', '-modinfo', help='print module info', action='store_const', const=True) 
parser.add_argument('-ptr', help='Print test results', action='store_const', const=True)
args = parser.parse_args()
print(args)

if __name__ == "__main__":
    if args.m:
       config.PRINT_MODULE_INFO = True
       import utilities.utils as utils
       utils.moduleinfo(locals())
    if args.ptr:
        #Update global printing variable
        config.PRINT_TEST_RESULTS = True
    if args.t:
        #Enable the cross module global testing
        config.RUN_ALL_NET_TESTS = True
        from test import test_ii as tii
        tii.run_tests()
