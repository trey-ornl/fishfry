#!/bin/bash
module load cpe/23.12
module load rocm/5.7.1
module load craype-accel-amd-gfx90a
module -t list
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
set -x
export ROCFFT_RTC_CACHE_PATH=/dev/null
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_VERSION_DISPLAY=1
TPN=8
TASKS=$(( $1 * $2 * $3 ))
NODES=$(( ( TASKS + TPN - 1 ) / TPN ))
ldd ./fishfry
srun -u -n ${TASKS} -N ${NODES} --gpus-per-node=8 --gpu-bind=closest --exclusive -t 5:00 ./fishfry $1 $2 $3 $4 $5 $6 $7
