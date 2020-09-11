from loss import triple_loss

from tqdm import tqdm, trange
from transformers import AdamW
import os
import torch
import logging

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

# todo: model_name_or_path is now output_dir?


def train(data_iterator, model, config):

    if args.local_rank in [-1, 0]:
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
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
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

    tr_loss, logging_loss = 0., 0.

    # set gradients of model parameters to zero
    model.zero_grad()

    train_iterator = trange(
        epochs_trained,
        config["num_train_epochs"],
        desc='Epoch',
        disable=args.local_rank not in [-1, 0], # todo: look into this
    )

    for iteration in train_iterator:
        epoch_iterator = tqdm(
            data_iterator,
            desc='Iteration',
            disable=args.local_rank not in [-1, 0],
        )
        for step, batch in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()

            batch = [t.to(config['device']) for t in batch]
            batch_input_ids = batch[0]
            batch_attention_mask = batch[1]
            batch_token_type_ids = batch[2]
            batch_y_start = batch[3]
            batch_y_end = batch[4]
            batch_y = batch[5]

            logits_start, logits_end, logits_class = model(
                batch_input_ids,
                batch_attention_mask,
                batch_token_type_ids,
            )

            loss = triple_loss(
                (logits_start, logits_end, logits_class),
                (batch_y_start, batch_y_end, batch_y),
            )

            # average over gpus
            if config['n_gpu'] > 1:
                loss = loss.mean()

            if config['gradient_accumulation_steps'] > 1:
                loss /= config['gradient_accumulation_steps']

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

                model.zero_grad() # todo: check orig code for when zero_grad is performed

                global_step += 1

                # todo: log training metrics
                # todo: eval

                # save model checkpoint
                if args.local_rank in [-1, 0] and \
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
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step
