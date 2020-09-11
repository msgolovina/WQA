import argparse
from pathlib import Path
import torch
import yaml

parser = argparse.ArgumentParser()

parser.add_argument(
    '--train-data-path', type=str, default='./data/train.jsonl', required=True
)
parser.add_argument(
    '--test-data-path', type=str,
)
parser.add_argument(
    '--log-dir', type=str, default='./logs', required=True
)
parser.add_argument(
    '--config-path', type=str, default='./training/config/base_config.yml'
)
parser.add_argument(
    "--fp16",
    action="store_true",
)
parser.add_argument(
    "--fp16-opt-level",
    type=str,
    default="O1",
    help="Apex amp optimization level in ['00', '01', '02', '03']."
         "Read more: https://nvidia.github.io/apex/amp.html",
)

if __name__ == '__main__':
    args = parser.parse_args()

    config = {}
    
    # device
    if torch.cuda.is_available():
        config['device'] = 'cuda'
        config['n_gpu'] = torch.cuda.device_count()
    else:
        config['device'] = 'cpu'
        config['n_gpu'] = 0
    # config['local_rank'] = -1

    # 16-bit mixed precision
    config['fp16'] = args.fp16
    config['fp16_opt_level'] = args.fp16_opt_level

    # paths

    config['train_data_path'] = args.train_data_path
    config['test_data_path'] = args.test_data_path
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    config['log_dir'] = str(log_dir)
    config['output_dir'] = str(log_dir / 'output')

    # seed
    config['seed'] = 777

    # training config
    config['pretrained_bert_name'] = 'bert-base-uncased'
    config['max_seq_len'] = 384
    config['num_labels'] = 5
    config['num_train_epochs'] = 2
    config['per_gpu_batch_size'] = 4
    config['batch_size'] = config['per_gpu_batch_size'] * max(1, config['n_gpu'])
    config['max_steps'] = -1
    config['learning_rate'] = 0.00005
    config['adam_epsilon'] = 1e-8
    config['weight_decay'] = 0.0
    config['max_grad_norm'] = 1.0

    with open(args.config_path, 'w') as f:
        yaml.dump(config, f)
