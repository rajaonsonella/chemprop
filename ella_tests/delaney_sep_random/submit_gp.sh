#!/bin/bash

#SBATCH --account=def-aspuru
#SBATCH --cpus-per-task=8
#SBATCH --time=0:40:00
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

python gp_model.py > gp.out 2>&1

echo FINISHED at `date`
