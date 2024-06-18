#!/bin/bash
source ./modules-frontier
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
set -x
export ROCFFT_RTC_CACHE_PATH=/dev/null
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_VERSION_DISPLAY=1
export JAWFISH_DHT_PER_RANK=4
export JAWFISH_EPS='1.1e-7'
export JAWFISH_MAX_OUTER_ITERS=21
export JAWFISH_SMOOTHER_RADIUS=5
export JAWFISH_SMOOTHER_ITERS=1
ulimit -c 0
srun -u -n 8 -N 1 --gpus-per-node=8 --gpu-bind=closest --exclusive -t 5:00 ./fishfry 2 2 2 128 128 128 
srun -u -n 16 -N 2 --gpus-per-node=8 --gpu-bind=closest --exclusive -t 5:00 ./fishfry 4 2 2 128 128 128 
srun -u -n 16 -N 2 --gpus-per-node=8 --gpu-bind=closest --exclusive -t 5:00 ./fishfry 2 4 2 128 128 128 
srun -u -n 16 -N 2 --gpus-per-node=8 --gpu-bind=closest --exclusive -t 5:00 ./fishfry 2 2 4 128 128 128 
srun -u -n 8 -N 1 --gpus-per-node=8 --gpu-bind=closest --exclusive -t 5:00 ./fishfry 1 4 2 128 128 128 
srun -u -n 8 -N 1 --gpus-per-node=8 --gpu-bind=closest --exclusive -t 5:00 ./fishfry 2 1 4 128 128 128 
#srun -u -n 8 -N 1 --gpus-per-node=8 --gpu-bind=closest --exclusive -t 5:00 ./fishfry 2 2 2 512 512 512
