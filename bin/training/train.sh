TRAIN_DATA_PATH=$TRAIN_DATA_PATH
TEST_DATA_PATH=$TEST_DATA_PATH
LOG_DIR=$LOG_DIR
CONFIG_PATH=${CONFIG_PATH:-"training/config/base_config.yml"}

PYTHONPATH=. python3 -m torch.distributed.launch --nproc_per_node 4 train.py \
  --do-train \
  --config-path=${CONFIG_PATH}
