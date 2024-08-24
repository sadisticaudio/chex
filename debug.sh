#!/bin/bash

# CXX=/usr/bin/clang++-10
# Torch_DIR=/media/frye/sda5/libtorch-2.0.1-RC
# CUDA_LAUNCH_BLOCKING=1
# TORCH_USE_CUDA_DSA=1
# CUDACXX=/usr/local/cuda-11.8/bin/nvcc
# export ASAN_OPTIONS=fast_unwind_on_malloc=false

CFGTYPE=DEBUG

mkdir -p build$CFGTYPE
cd build$CFGTYPE
cmake -DCMAKE_BUILD_TYPE=$CFGTYPE ..
cmake --build . --config $CFGTYPE --target cpptest
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'/media/frye/sda5/boost_1_85_0/stage/lib:/media/frye/sda5/anaconda3/envs/pytorch_env/lib/python3.10/site-packages/torch/lib'
gdb ./cpptest
cd ..
