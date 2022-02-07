#!/bin/bash
module load rocm
module load craype-accel-amd-gfx90a
module list
set -x
export CXX=hipcc
export CXXFLAGS="-DO_HIP -g -O -std=c++11 --offload-arch=gfx90a -Wall -I${CRAY_MPICH_PREFIX}/include"
export LD=CC
export LDFLAGS="-DO_HIP -g -O -std=c++11 -Wall -L${ROCM_PATH}/lib"
export LIBS='-lhipfft -lamdhip64 -lhsa-runtime64'
make clean
make -j
