#!/bin/bash

for dir in *split*; do
    echo $dir
    cd $dir
    for subdir in *ratio*; do
        echo $subdir
        cd $subdir
        for f in *submit*; do
            #echo $f
            sbatch $f
        done
        cd ..
    done
    cd ..
done
