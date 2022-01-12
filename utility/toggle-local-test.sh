#!/bin/bash

function local-test-on {
    #manual local testing vs nightly automatic testing
    local pyfile=$1
    ([[ -f $pyfile ]] && [[ $pyfile = *.py ]] ) && sed 's/local_test = False/local_test = True/' <$pyfile
    return 0
}

function local-test-off {
    local pyfile=$1
    ([[ -f $pyfile ]] && [[ $pyfile = *.py ]] ) && sed 's/local_test = True/local_test = False/' <$pyfile
    return 0
}

test_dir=$1
[[ -d $1 ]] || ( echo "$1 is not a directory" && exit 1)

switch=$2

[[ $switch != "-on" || $switch != "-off" ]] || ( echo "$switch must be '-on' or '-off'" && exit 1)

function toggle {
    local switch=$1
    if [[ $switch = "-on" ]]; then
	    func=local-test-on
    elif [[ $siwtch = "-off" ]]; then
	    func=local-test-off
    else
	    echo "$switch != -on or -off" && exit 1
    fi
    return 0
}

for file in $(ls $test_dir); do
    $func $test_dir/$file
done

