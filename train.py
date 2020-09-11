from dataset import NQDataset
from train_function import train
from evaluate_function import evaluate
from models import BertForQA
from utils import set_seed

import argparse
import json
import logging
import os
import yaml
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, BertConfig
import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config-path', type=str, default='./training/config/base_config.yml'
)
parser.add_argument(
    '--do-train', action='store_true'
)
parser.add_argument(
    "--local_rank",
    type=int,
    default=-1,
    help="local_rank for distributed training on gpus",
)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    tokenizer = BertTokenizer.from_pretrained(
        config['pretrained_bert_name'],
        do_lower_case=True
    )

    # setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN
    )

    logger.warning(
        'Process rank: %s, device: %s, '
        'n_gpu: %s, distributed training: %s, 16-bits training: %s',
        args.local_rank,
        config['device'],
        config['n_gpu'],
        bool(args.local_rank != -1),
        config['fp16'],
    )

    set_seed(config['seed'], config['n_gpu'])

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    bert_config = BertConfig.from_pretrained(config['pretrained_bert_name'])
    bert_config.num_labels = config['num_labels']
    model = BertForQA.from_pretrained(config['pretrained_bert_name'], config=bert_config)
    # todo: torch distrib barrier?
    model.to(config['device'])

    logger.info(f'Training/eval params: {config}')

    if args.do_train:
        logger.info('Uploading training file %s', config['train_data_path'])
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
        logger.info(
            ' global_step = %s, average loss = %s',
            global_step,
            tr_loss
        )
        if args.do_train and (
                args.local_rank == -1 or torch.distributed.get_rank() == 0):

            if not os.path.exists(config['output_dir']):
                os.makedirs(config['output_dir'])

            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(config['output_dir'])
            tokenizer.save_pretrained(config['output_dir'])
            # model = BertForQA.from_pretrained(config['output_dir']) todo: why tho
            #model.to(config['device']) todo: why tho

            # Evaluation # todo: optional eval for all checkpoints
            results = {}
            if args.do_eval and args.local_rank in [-1, 0]:
                # if args.do_train:
                logger.info(
                        "Loading checkpoints saved during training")
                checkpoints = [config['output_dir']]
                # else:
                    # logger.info("Loading checkpoint %s for evaluation",
                    #             args.model_name_or_path)
                    # checkpoints = [args.model_name_or_path] # todo: check how this and out_dir are different

                logger.info("Evaluate the following checkpoints: %s",
                            checkpoints)
                for checkpoint in checkpoints:
                    global_step = checkpoint.split("-")[-1] if len(
                        checkpoints) > 1 else ""
                    model = BertForQA.from_pretrained(
                        checkpoint)  # , force_download=True)
                    model.to(args.device)

                    result = evaluate()#args, model, tokenizer,
                                      #prefix=global_step) # todo: add evaluation function
                    result = dict((k + (
                        "_{}".format(global_step) if global_step else ""), v)
                                  for k, v in result.items())
                    results.update(result)

            logger.info("Results: {}".format(results))
            with open(os.path.join(config['output_dir'], 'results.json'), 'w'):
                json.dump(results)
