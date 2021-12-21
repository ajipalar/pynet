#!/usr/bin/env python3
"""
A small python module for configuring pynet testing
"""
from pathlib import Path
import sys

def impure_pynet_testing_path_setup():
    pynet_home = Path('.').absolute().parent
    sys.path.append(str(pynet_home))
