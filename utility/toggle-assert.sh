#!/bin/bash

function assert_off {
  #usage: $1 is a python file
  local pyfile=$1
  [[ "$pyfile" == *.py ]] && echo "assert off $pyfile"
  return 0
}

function assert_on {
  #usage: $1 is a python file
  local pyfile=$1
  [[ "$pyfile" == *.py ]] && echo "assert on $pyfile"
  return 0
}

function file_loop {
  local flag=$1
  case $flag in
    -on)
      local toggler=assert_on 
      ;;
    -off)
      local toggler=assert_off 
      ;;
    *)
      echo "invalid flag"
      echo 'usage: "-on", "-off"'
      exit 1
      ;;
  esac
  local files=$(ls $2)
  for file in $files; do
    $toggler $file
  done
}

file_loop $1 $2
