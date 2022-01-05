#!/bin/bash
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

input=$(echo $1 | sed 's/\/$//')

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

exit 1
