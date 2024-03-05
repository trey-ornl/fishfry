#!/bin/bash
module load cuda/11.0.3
module -t list
set -x
export CXX=nvcc
export CXXFLAGS="-x cu -arch=sm_60 -std=c++17 -g -O3 --expt-extended-lambda"
#export CXXFLAGS="-DPARIS_5PT ${CXXFLAGS}"
#export CXXFLAGS="-DPARIS_3PT ${CXXFLAGS}"
export LD=mpiCC
export LDFLAGS="-g -O3 -L${CUDA_DIR}/lib64"
export LIBS='-lcudart -lcufft'
make clean
make
