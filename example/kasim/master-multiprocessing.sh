#!/bin/sh
export PYTHONPATH="$PYTHONPATH:$HOME/opt/github-repositories/glucksfall.sterope"

MODEL=pysbmodel-example6-kasim.kappa
FINAL=660
STEPS=10

NUM_SIMS=1
NUM_LEVELS=1000

python3 -m sterope.kasim --model=$MODEL --final=$FINAL --steps=$STEPS \
--sims=$NUM_SIMS --grid=$NUM_LEVELS --tmin=600 --type global --syntax=3 --kasim=kasim4-rc1 --python /usr/bin/python3
