#!/bin/bash

#Usage: $1 -w or -c
#     : $2 directory or file

function mysha256 {
  # dir must be a directory
  local dir=$1
  [[ -d $dir ]] || exit 1

  local files=$(ls $dir)
  for file in $files; do
    if [[ -f $dir/$file ]]; then
      shasum -a 256 $dir/$file
    fi
  done
  return 0
}

function pynet_checksum {
  # dir must be a directory
  local dir=$1
  [[ -d $dir ]] || exit 1
  #shasum -q -c $dir/checksumfile || exit 1
  shasum -c $dir/checksumfile || (echo "Checksum failed" && exit 1)
}

function write_checksum_file {
  local input=$1
  if [[ -d $input ]]; then
    mysha256 $input > $input/checksumfile
    sed '/checksumfile/d' <$input/checksumfile > $input/checksumfile.tmp
    cat $input/checksumfile.tmp > $input/checksumfile
    rm $input/checksumfile.tmp
    exit 0
  elif [[ -f $input ]]; then
    shasum -a 256 $input
    exit 0
  else
    echo "Error"
    echo "The first positional argument '$input'"
    echo "is neither a file nor a director"
    exit 1
  fi
}

# Begin

[[ $1 = '-w' || $1 = '-c' ]] || (echo 'Invalid input: use -c, -w' && exit 1)
input=$(echo $2 | sed 's/\/$//')

if [[ $1 = '-w' ]]; then
  write_checksum_file $input
  exit 0
elif [[ $1 = '-c' ]]; then
  pynet_checksum $input
  exit 0
else
  echo "Error: branch shouldn't be executed"
  exit 1
fi

echo "Error: program should terminate sooner"
exit 1
