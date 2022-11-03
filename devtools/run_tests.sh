#!/bin/bash
module load sali
conda activate pynetp3.10.0

pwd
prefix='test/local/'
for pytest in $(ls $prefix'test_'*.py); do
	string=${pytest%.py}
	module=${string#"$prefix"}
	echo "Running $module"
	python3 -m 'test.local.'$module
done
