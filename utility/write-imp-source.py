"""
Dev tool to save you some typing when writing imp testdefs code

inputs foo.h or foo.cpp
oputputs

writes the header guard
writes the namespace
if the header exists in foo/include or foo/include/internal
writes and file is foo.cpp writes the file and function
signatures
"""

import re
import argparse
from pathlib import Path
from typing import NewType

FilePath = NewType('FilePath', Path)
DirPath = NewType('DirPath', Path)

def check_imp_dir_structure():
    pass

def is_h(f: FilePath) -> bool:
    return f.is_suffix() == '.h'

def is_cpp(f: FilePath) -> bool:
    return f.is_suffix() == '.cpp'

def is_exists(f: FilePath) -> bool:
    return f.is_file()

def is_imp_module(wd: DirPath) -> bool:
    #All conditions must pass
    answer = True
    if not (wd / 'test').exists() : answer = False
    if not (wd / 'pyext').exists() : answer = False
    if not (wd / 'src').exists() : answer = False
    if not (wd / 'utility').exists() : answer = False
    if not (wd / 'examples').exists() : answer = False
    if not (wd / 'examples').exists() : answer = False

    return answer


def get_module_name(wd: DirPath) -> str:
    return mod_name
