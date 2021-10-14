#!/bin/bash
module load rocm
module load craype-accel-amd-gfx908
module list
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
set -x
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_VERSION_DISPLAY=1
NTASKS=$(( $1 * $2 * $3 ))
srun -u -n ${NTASKS} --ntasks-per-node=4 --exclusive -t 5:00 ./fishfry $1 $2 $3 $4 $4 $4 $5
