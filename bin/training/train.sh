TRAIN_DATA_PATH=$TRAIN_DATA_PATH
TEST_DATA_PATH=$TEST_DATA_PATH
LOG_DIR=$LOG_DIR
CONFIG_PATH=${CONFIG_PATH:-"training/config/base_config.yml"}

PYTHONPATH=. python train.py \
  --do-train \
  --config-path=${CONFIG_PATH}