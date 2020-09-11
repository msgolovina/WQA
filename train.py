from dataset import NQDataset
from train_function import train
from models import BertForQA

import argparse
import logging
import os
import yaml
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, BertConfig

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config-path', type=str, default='./training/config/base_config.yml'
)
parser.add_argument(
    '--do-train', action='store_true'
)


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    tokenizer = BertTokenizer.from_pretrained(
        config['pretrained_bert_name'],
        do_lower_case=True
    )
    bert_config = BertConfig.from_pretrained(config['pretrained_bert_name'])
    bert_config.num_labels = config['num_labels']
    model = BertForQA.from_pretrained(config['pretrained_bert_name'], config=bert_config)
    model.to(config['device'])

    logger.info(f'Training/eval params: {config}')

    if args.do_train:
        batch_size = config['per_gpu_batch_size'] * max(1, config['n_gpu'])
        dataset = NQDataset(
            config['train_data_path'],
            tokenizer,
        )
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler,
                                      batch_size=batch_size,
                                      pin_memory=True)
        global_step, tr_loss = train(
            dataloader, model, config
        )
        if not os.path.exists(config['output_dir']):
            os.makedirs(config['output_dir'])

        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(config['output_dir'])
        tokenizer.save_pretrained(config['output_dir'])
        model = BertForQA.from_pretrained(config['output_dir'])
        model.to(config['device'])

        # todo: loading from checkpt
        # todo: add eval
        # todo: add metrics
