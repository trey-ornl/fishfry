#!/bin/bash
module load rocm
module load craype-accel-amd-gfx908
module list
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
set -x
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=NUMA
export MPICH_VERSION_DISPLAY=1
TPN=8
TASKS=$(( $1 * $2 * $3 ))
NODES=$(( ( TASKS + TPN - 1 ) / TPN ))
CORES=$(( 64 / TPN ))
srun -u -n ${TASKS} -N ${NODES} -c ${CORES} --gpus-per-node=8 --gpu-bind=closest --exclusive -t 5:00 ./fishfry $1 $2 $3 $4 $4 $4 $5
