#!/bin/bash
#SBATCH --exclusive
#SBATCH -t 5:00
#SBATCH -o %x-%j.out
module load rocm
module load craype-accel-amd-gfx90a
module -t list
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
set -x
export ROCFFT_RTC_CACHE_PATH=/dev/null
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_VERSION_DISPLAY=1
TASKS=$(( $1 * $2 * $3 ))
srun -n ${TASKS} --gpus-per-node=8 --gpu-bind=closest --exclusive -t 5:00 ./fishfry $1 $2 $3 $4 $5 $6 $7
