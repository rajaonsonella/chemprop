#!/bin/bash

#SBATCH --account=def-aspuru
#SBATCH --cpus-per-task=8
#SBATCH --time=1:40:00
#SBATCH --mem-per-cpu=8G
#SBATCH --output=baseline.slurm

echo STARTED at `date`

echo Activate Environment

PWD=`pwd`
echo $PWD

activate () {
  . /home/rajao/pdm/bin/activate
    }

activate

echo Load modules

module restore my_modules

echo Start running

python chemprop_model.py > chemprop.out 2>&1

echo FINISHED at `date`
