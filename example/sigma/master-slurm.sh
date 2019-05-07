#!/bin/sh

#SBATCH --no-requeue
#SBATCH --partition=spica

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --job-name=sterope
#SBATCH --output=stdout.txt
#SBATCH --error=stderr.txt

<<<<<<< HEAD
export PYTHONPATH="$PYTHONPATH:$HOME/opt/github-repositories/glucksfall.sterope"

MODEL=pysbmodel-example6-kasim.kappa
FINAL=660
STEPS=10

NUM_LEVELS=1

python3 -m sterope.kasim --model=$MODEL --final=$FINAL --steps=$STEPS \
--grid=$NUM_LEVELS --tmin=600 --type global --syntax=3 \
--kasim=kasim4-rc1 --python /usr/bin/python3 --slurm=$SLURM_JOB_PARTITION
=======
export PYTHONPATH="$PYTHONPATH:$HOME/opt/github-repositories/sterope.glucksfall"

MODEL=model.kappa
FINAL=90
STEPS=10

NUM_LEVELS=1000

python3 -m sterope.kasim --model=$MODEL --final=$FINAL --steps=$STEPS \
--grid=$NUM_LEVELS --type global --syntax=3 \
--kasim=kasim4 --python /usr/bin/python3
>>>>>>> 258e7d9c80b7bb1e13219585d60827e1f5381e87
