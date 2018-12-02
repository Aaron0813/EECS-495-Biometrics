#!/usr/bin/env bash
for img in `ls *.jpg`
do
    convert -rotate 270 $img $img
done
