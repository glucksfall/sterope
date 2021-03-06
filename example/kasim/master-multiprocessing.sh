#!/bin/sh
export PYTHONPATH="$PYTHONPATH:$HOME/opt/github-repositories/sterope.glucksfall"

MODEL=pysbmodel-example6-kasim.kappa
FINAL=660
STEPS=10

NUM_LEVELS=1

python3 -m sterope.kasim --model=$MODEL --final=$FINAL --steps=$STEPS \
--grid=$NUM_LEVELS --tmin=600 --type global --syntax=3 \
--kasim=kasim4-rc1 --python /usr/bin/python3
