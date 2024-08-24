#!/bin/bash

CFGTYPE=RELEASE
[[ $1 ]] && CFGTYPE=$1
./build.sh $CFGTYPE
cp ./cimporter.py ./build$CFGTYPE/
eval "$(conda shell.bash hook)"
# conda activate pytorch_env
conda activate jupyter_env
cd ./build$CFGTYPE
# CUDA_LAUNCH_BLOCKING=1 LD_PRELOAD="/lib/x86_64-linux-gnu/libasan.so.8" 
python cimporter.py
# conda deactivate
# cd ../
