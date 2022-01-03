#!/bin/bash
sed '/assert / s/^#\([^#]\)\(.*\)/\1\2/' <$1 >$1.mytmp
cat $1.mytmp > $1
rm $1.mytmp
