#!/bin/bash

for f in ../all/task1/*
do 
    if [[ $f == *"train-$1"* ]]; then
        python run.py --train $f --dev ${f/train-$1/dev} -c 100 -C 200 -d $2 -b 32
    fi
done