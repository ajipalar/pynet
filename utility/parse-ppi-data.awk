#!/bin/sh

path="${2:-pyext/data/sars-cov-2-ppi/41586_2020_2286_MOESM5_ESM.csv}"
column=$1
echo $path
awk 'BEGIN { FS="\t";}

{ print $'$column' }' < $path
