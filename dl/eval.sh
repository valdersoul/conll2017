#!/bin/bash

for f in ../all/task1/*
do 
    if [[ $f == *"train-$1"* ]]; then
        base=${f##*/}
        python run.py --train $f --test ${f/train-$1/dev} -e 1 -r ../model/train-high-models/model$base.pkl -R ../results/$base-out -l $1.log >> temp.txt
        rm temp.txt
    fi
done