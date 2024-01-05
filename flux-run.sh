#!/bin/bash
module load rocm
module load craype-accel-amd-gfx90a
module -t list
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
set -x
export ROCFFT_RTC_CACHE_PATH=/dev/null
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_VERSION_DISPLAY=1
TPN=8
TASKS=$(( $1 * $2 * $3 ))
NODES=$(( ( TASKS + TPN - 1 ) / TPN ))
ldd ./fishfry
flux run --exclusive -t 5 -N ${NODES} -n ${TASKS} -c 8 -g 1 -o gpu-affinity=per-task ./fishfry $1 $2 $3 $4 $5 $6 $7
