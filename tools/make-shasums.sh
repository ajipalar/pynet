#!/bin/bash
for file in "$1"/*
do
shasum -a 256 $file
done
