#!/bin/bash

output_prefix="[updatenotebooks]"

echo "$output_prefix updating paired notebooks"
for notebook in $( ls *.ipynb);
do
        strlen=${#notebook}
        suffend=$strlen-1
        suffbegin=$strlen-5
        source_file="${notebook:0:$suffbegin-1}.py"
        # jupytext --update --to notebook $source_file
        jupytext --sync $notebook


done

echo "$output_prefix done"


