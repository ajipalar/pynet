#!/bin/bash
for file in $(ls .checksums); do
	shasum -c .checksums/$file
done
