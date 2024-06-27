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
export JAWFISH_SMOOTHER_RADIUS=10
export JAWFISH_SMOOTHER_ITERS=1
ulimit -c 0
srun -u -n 1 -N 1 --gpus-per-task=1 --gpu-bind=closest --exclusive -t 5:00 ./fishfry 1 1 1 64 64 64
