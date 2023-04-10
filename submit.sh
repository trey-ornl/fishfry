#!/bin/bash
TPN=8
TASKS=$(( $1 * $2 * $3 ))
NODES=$(( ( TASKS + TPN - 1 ) / TPN ))
NAME="fishfry-$1x$2x$3-$4x$5x$6"
sbatch -J ${NAME} -N ${NODES} job.sh $1 $2 $3 $4 $5 $6 $7

