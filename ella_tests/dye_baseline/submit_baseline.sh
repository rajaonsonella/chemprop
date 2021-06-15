#!/bin/bash

#SBATCH --account=def-aspuru
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --output=baseline.slurm

echo STARTED at `date`

echo Activate Environment

cd /home/rajao/chemprop/ella_tests/dye_baseline

PWD=`pwd`
echo $PWD

activate () {
  . /home/rajao/pdm/bin/activate
    }

activate

echo Load modules

module restore my_modules

echo Start running
python run_baseline.py > r_baseline.out 2>&1

echo FINISHED at `date`
