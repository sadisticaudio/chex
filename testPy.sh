#!/bin/bash

CFGTYPE=RELEASE
[[ $1 ]] && CFGTYPE=$1
./build.sh $CFGTYPE
cp cimporter.py src/chex/lib/
eval "$(conda shell.bash hook)"
conda activate pytorch_env
cd src/chex/lib
python cimporter.py
cd ../../../
conda deactivate
