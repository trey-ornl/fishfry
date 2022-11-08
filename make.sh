#!/bin/bash
module load rocm
module load craype-accel-amd-gfx90a
module list
set -x
export CXX=hipcc
export CXXFLAGS="-DO_HIP -g -O3 -std=c++11 --offload-arch=gfx90a -Wall -I${CRAY_MPICH_PREFIX}/include"
#export CXXFLAGS="-DPARIS_5PT ${CXXFLAGS}"
#export CXXFLAGS="-DPARIS_3PT ${CXXFLAGS}"
export LD=CC
export LDFLAGS="-DO_HIP -g -O3 -std=c++11 -Wall -L${ROCM_PATH}/lib"
export LIBS='-lhipfft -lamdhip64'
make clean
make #-j
