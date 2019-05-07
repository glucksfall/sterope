#!/bin/sh

#SBATCH --no-requeue
#SBATCH --partition=spica

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --job-name=sterope
#SBATCH --output=stdout.txt
#SBATCH --error=stderr.txt

export PYTHONPATH="$PYTHONPATH:$HOME/opt/github-repositories/sterope.glucksfall"

MODEL=model.kappa
FINAL=90
STEPS=10

NUM_LEVELS=1000

python3 -m sterope.kasim --model=$MODEL --final=$FINAL --steps=$STEPS \
--grid=$NUM_LEVELS --type global --syntax=3 \
--kasim=kasim4 --python /usr/bin/python3
