#!/bin/bash
function mysha256 {
  local dir=$1
  if [[ -d $dir ]]; then
    for file in "$dir"/*
    do
      if [[ -f $file ]]; then
      shasum -a 256 $file
      fi
    done
  fi
  return 0
}

if [[ -d $1 ]]; then
  mysha256 $1
elif [[ -f $1 ]]; then
  shasum -a 256 $1
else
  echo "Error"
  echo "The first positional argument '$1'"
  echo "is neither a file nor a director"
  exit 1
fi

exit 0
