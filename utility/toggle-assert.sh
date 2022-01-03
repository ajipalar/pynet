#!/bin/bash

function assert_off {
  #usage: $1 is a python file
  local pyfile=$1
  if [[ "$pyfile" == *.py ]]; then
    echo "assert off $pyfile"
    sed '/assert / s/\(^[^#]\)\(.*\)/#\1\2/' <$pyfile >$pyfile.mytmp
    cat $pyfile.mytmp > $pyfile
    rm $pyfile.mytmp
  fi
  return 0
}

function assert_on {
  #usage: $1 is a python file
  local pyfile=$1
  if [[ "$pyfile" == *.py ]]; then
    echo "assert on $pyfile"
    sed '/assert / s/^#\([^#]\)\(.*\)/\1\2/' <$pyfile >$pyfile.mytmp
    cat $pyfile.mytmp > $pyfile
    rm $pyfile.mytmp
  fi 
  return 0
}

function file_loop {
  #usage: $1 is a flag
  #       $2 is a directory
  local toggler=$1
  local dir=$2
  [[ -d $dir ]] && local files=$(ls $dir)
  for file in $files; do
    $toggler $dir/$file
  done
  return 0
}

function toggler_dispatcher {
  local flag=$global_toggle_flag
  case $flag in
    -on)
      global_toggler_function=assert_on 
      ;;
    -off)
      global_toggler_function=assert_off 
      ;;
    *)
      echo "invalid flag"
      echo 'usage: "-on", "-off"'
      exit 1
      ;;
  esac
  return 0
}

global_toggle_flag=$1
global_operand=$2
#sets the global_toggler_function variable to assertions on or off
toggler_dispatcher $global_toggle_flag

if [[ -d $global_operand ]]; then
  file_loop $global_toggler_function $global_operand
  exit 0
elif [[ -f $global_operand ]]; then
  $global_toggler_function $global_operand
  exit 0
else
  echo "The global operand '$global_operand'"
  echo "is neither a file nor a directory"
  echo "exiting"
  exit 1
fi

echo "Error: never exited after if statements"
exit 1

#asserts on
sed '/assert / s/^#\([^#]\)\(.*\)/\1\2/' <$1 >$1.mytmp
cat $1.mytmp > $1
rm $1.mytmp

#asserts off


