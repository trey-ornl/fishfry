#!/bin/bash
#BSUB -alloc_flags smt1
#BSUB -W 5
#BSUB -J fishfry

module load cuda/11.0.3
module -t list
set -x
ldd ./fishfry
date
jsrun -a1 -c7 -g1 -r6 -brs --smpiargs="-gpu" ./fishfry 1 2 3 128 256 384 10

