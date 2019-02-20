#!/bin/sh
export PYTHONPATH="$PYTHONPATH:$HOME/opt/github-repositories/glucksfall.sterope"

MODEL=pysbmodel-example6-kasim.kappa
FINAL=660
STEPS=10

NUM_SIMS=10
NUM_LEVELS=10

python3 -m sterope.kasim --model=$MODEL --final=$FINAL --steps=$STEPS \
--sims=$NUM_SIMS --syntax=3 --seed=0 --kasim=kasim4-rc1 --python /usr/bin/python3
