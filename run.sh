#!/bin/bash

CFGTYPE=RELEASE
[[ $1 ]] && CFGTYPE=$1
# ./build.sh $CFGTYPE
./testPy.sh $CFGTYPE
