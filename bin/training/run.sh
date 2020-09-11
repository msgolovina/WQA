#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

# usage:
# DATA_DIR=... \
# LOG_DIR=... \
# ...
#
# bash -x bin/run.sh

TRAIN_DATA_PATH=$TRAIN_DATA_PATH
TEST_DATA_PATH=$TEST_DATA_PATH
LOG_DIR=$LOG_DIR
CONFIG_PATH=${CONFIG_PATH:-"training/config/base_config.yml"}

echo "-------------------------"
echo "RUN SETUP"
echo "-------------------------"
echo "TRAIN_DATA_PATH: $TRAIN_DATA_PATH"
echo "TEST_DATA_PATH: $TEST_DATA_PATH"
echo "LOG_DIR: $LOG_DIR"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "-------------------------"

mkdir -p ${LOG_DIR}

echo "PREPARING CONFIG"
PYTHONPATH=. python scripts/prepare_config.py \
  --train-data-path=${TRAIN_DATA_PATH} \
  --log-dir=${LOG_DIR} \
  --config-path=${LOG_DIR}/config.yml \
  --fp16

echo "TRAINING"
CONFIG_PATH=${LOG_DIR}/config.yml bash -x bin/training/train.sh

if [[ -f "$TEST_DATA_PATH"]]; then
  filename = $(basename $TEST_DATA_PATH)
  echo "EVALUATING $filename"
  PYTHONPATH=. python evaluate.py \
    --in-csv=${TEST_DATA_PATH} \
    --out-dir=${LOG_DIR}/predictions/${filename} \
    --model=${LOG_DIR}/checkpoints/best.traced.cpu.pth
fi

