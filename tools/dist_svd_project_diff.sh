#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT  \
    $(dirname "$0")/analysis_tools/svd_project_difference_covariance.py $CONFIG  \
      --seed 0  --launcher pytorch ${@:3}
