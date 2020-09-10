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

if __name__ == '__main__':
    args = parser.parse_args()

    params = {}

    # paths

    params['train_data_path'] = args.train_data_path
    params['test_data_path'] = args.test_data_path
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    params['log_dir'] = str(log_dir)
    params['output_dir'] = str(log_dir / 'output')

    # training params

    params['seed'] = 777
    params['pretrained_bert_name'] = 'bert-base-uncased'
    params['max_seq_len'] = 384
    params['num_labels'] = 5
    params['num_train_epochs'] = 2
    params['learning_rate'] = 0.00005
    params['adam_epsilon'] = 1e-8
    params['weight_decay'] = 0.0
    params['max_grad_norm'] = 1.0
    params['local_rank'] = -1
    params['per_gpu_batch_size'] = 4
    params['max_steps'] = -1

    # device

    if torch.cuda.is_available():
        params['device'] = 'cuda'
        params['n_gpu'] = torch.cuda.device_count()
    else:
        params['device'] = 'cpu'
        params['n_gpu'] = 0

    with open(args.config_path, 'w') as f:
        yaml.dump(params, f)
