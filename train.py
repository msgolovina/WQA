from dataset import NQDataset
from evaluate_function import evaluate
from loss import triple_loss
from models import BertForQA
from utils import set_seed

import argparse
import json
import logging
import os
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, BertTokenizer, BertConfig
import torch
from tqdm import tqdm, trange
import yaml

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def train(data_iterator, model, config):

    if config['local_rank'] in [-1, 0]:
        tb_writer = SummaryWriter()

    if config['max_steps'] > 0:
        num_train_steps = config['max_steps']
        num_train_epochs = config['max_steps'] // (
            len(data_iterator) // config['gradient_accumulation_steps']) + 1
    else:
        num_train_steps = len(data_iterator) // \
                      config['gradient_accumulation_steps'] * \
                      config['num_train_epochs']

    no_decay = ['bias', 'LayerNorm.weight']

    # init optimizer with grouped parameters; init scheduler
    optimizer_params = [
        {
            'weight_decay': 0.0,
            'params': [
                param for name, param in model.named_parameters()
                if any(nd_name in name for nd_name in no_decay)
            ]
        },
        {
            'weight_decay': config['weight_decay'],
            'params': [
                param for name, param in model.named_parameters()
                if not any(nd_name in name for nd_name in no_decay)
            ]
        }
    ]

    optimizer = AdamW(
        optimizer_params,
        lr=config['learning_rate'],
        eps=config['adam_epsilon']
    )
    optimizer_path = os.path.join(config['output_dir'], 'optimizer.pt')
    if os.path.isfile(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path))

    if config['fp16']:
        try:
            from apex import amp
        except ImportError:
            raise ImportError
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=config['fp16_opt_level']
        )

    # parallelize model if multiple GPUs are available
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # distributed training
    if config['local_rank'] != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config['local_rank']],
            output_device=config['local_rank'],
            find_unused_parameters=True,
        )

    # training
    logger.info('==========================')
    logger.info('---------TRAINING---------')
    logger.info('==========================')
    logger.info(f' Num epochs: {config["num_train_epochs"]}')
    logger.info(f' Batch size per GPU: {config["per_gpu_batch_size"]}')
    logger.info(f'Num training steps: {num_train_steps}')

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0 # todo: look into it in the orig code; do I have to change it somehow later?

    if os.path.exists(config['output_dir']):
        try:
            logger.info('Uploading last checkpoint')
            suffix = config['output_dir'].split('-')[-1].split('/')[0]
            global_step = int(suffix)
            epochs_trained = global_step // \
                             (len(data_iterator) //
                              config['gradient_accumulation_steps'])
        except ValueError:
            logger.info('Fine-tuning pretrained model')

    logger.info(f'Setting tr_loss and logging_loss to zero')
    tr_loss, logging_loss = 0., 0.

    # set gradients of model parameters to zero
    model.zero_grad()

    train_iterator = trange(
        epochs_trained,
        config["num_train_epochs"],
        desc='Epoch',
        disable=config['local_rank'] not in [-1, 0], # todo: look into this
    )

    for iteration in train_iterator:
        epoch_iterator = tqdm(
            data_iterator,
            desc='Iteration',
            disable=config['local_rank'] not in [-1, 0],
        )
        for step, batch in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()

            logger.info(f'Reading batch')

            batch = [t.to(config['device']) for t in batch]
            batch_input_ids = batch[0]
            batch_attention_mask = batch[1]
            batch_token_type_ids = batch[2]
            batch_y_start = batch[3]
            batch_y_end = batch[4]
            batch_y = batch[5]

            logger.info(f'Batch len: {len(batch)}')

            logger.info(f'Forward pass:')
            logits_start, logits_end, logits_class = model(
                batch_input_ids,
                batch_attention_mask,
                batch_token_type_ids,
            )
            logger.info(f'Loss:')
            loss = triple_loss(
                (logits_start, logits_end, logits_class),
                (batch_y_start, batch_y_end, batch_y),
            )

            # average over gpus
            if config['n_gpu'] > 1:
                loss = loss.mean()

            if config['gradient_accumulation_steps'] > 1:
                loss /= config['gradient_accumulation_steps']

            logger.info(f'Backward pass:')
            if config['fp16']:
                with amp.scale_loss(loss, optimizer) as sc_loss:
                    sc_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % config['gradient_accumulation_steps'] == 0:
                if config['fp16']:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer),
                        config['max_grad_norm']
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config['max_grad_norm']
                    )
                optimizer.step()

                model.zero_grad()

                global_step += 1

                # todo: log training metrics
                # todo: eval

                # save model checkpoint
                if config['local_rank'] in [-1, 0] and \
                        config['save_steps'] > 0 and \
                        global_step % config['save_steps'] == 0:

                    output_dir = os.path.join(
                        config['output_dir'],
                        f'checkpoint-{global_step}'
                    )
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(output_dir)

                    torch.save(optimizer.state_dict(), optimizer_path)
                    logger.info('Saving model checkpoint and'
                                f'optimizer state to {output_dir}')

            if 0 < config['max_steps'] < global_step:
                epoch_iterator.close()
                break
        if 0 < config['max_steps'] < global_step:
            train_iterator.close()
            break
    if config['local_rank'] in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


if __name__ == '__main__':
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
    parser.add_argument("--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available")
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        config['n_gpu'] = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        config['n_gpu'] = 1
    config['device'] = device

    config['local_rank'] = args.local_rank
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
    if args.local_rank == 0:
        torch.distributed.barrier()
    model.to(config['device'])

    logger.info(f'Training/eval params: {config}')

    if args.do_train:
        logger.info('Uploading training file %s', config['train_data_path'])
        batch_size = config['per_gpu_batch_size'] * max(1, config['n_gpu'])
        dataset = NQDataset(
            config['train_data_path'],
            tokenizer,
        )
        sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
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
            model = BertForQA.from_pretrained(config['output_dir']) # todo: why tho
            model.to(config['device']) #  todo: why tho

            # Evaluation # todo: optional eval for all checkpoints

            if args.do_eval and args.local_rank in [-1, 0]:
                results = {}
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
                with open(os.path.join(config['output_dir'], 'results.json'), 'w') as f:
                    json.dump(results, f)
