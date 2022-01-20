#!/bin/bash

conda activate pynetp3.10.0

for test_file in $(ls "$1"); do
	( [[ -f "$1"/$test_file ]] && [[ "$1"/$test_file = *test*.py ]] ) || exit 1 
        test_file = sed 's/\(.*\)\.py/\1/'
        py  -m "test/$test_file"
