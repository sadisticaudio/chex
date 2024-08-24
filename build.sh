#!/bin/bash

CFGTYPE=RELEASE
[[ $1 ]] && CFGTYPE=$1

mkdir -p build$CFGTYPE
cd build$CFGTYPE
eval "$(conda shell.bash hook)"
conda activate pytorch_env
cmake -DCMAKE_BUILD_TYPE=$CFGTYPE -DCMAKE_PREFIX_PATH=$CONDA_PREFIX ..
cmake --build . --config $CFGTYPE
cd ..
# cp build$CFGTYPE/libChex.a lib/
# python -m build -s -w
conda deactivate
