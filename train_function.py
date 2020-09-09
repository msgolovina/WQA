from utils import set_seed
from loss import triple_loss

from tqdm import tqdm, trange
from transformers import AdamW
import os
import torch
import logging


logger = logging.getLogger(__name__)


def train(data_iterator, model, config):

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

    # training
    logger.info('==========================')
    logger.info('---------TRAINING---------')
    logger.info('==========================')
    logger.info(f' Num epochs: {config["num_train_epochs"]}')
    logger.info(f' Batch size per GPU: {config["per_gpu_batch_size"]}')

    # todo: upload checkpoint if exists

    global_step = 1
    epochs_trained = 0
    tr_loss, logging_loss = 0., 0.

    model.zero_grad()

    train_iterator = trange(epochs_trained,
                            config["num_train_epochs"],
                            desc='Epoch',
    )

    set_seed(config['seed'], config['n_gpu'])

    for iteration in train_iterator:

        epoch_iterator = tqdm(
            data_iterator,
            desc='Iteration',
        )

        for step, batch in enumerate(epoch_iterator):

            # todo: skip past trained steps if resuming training

            model.train()

            batch = [t.to(config['device']) for t in batch]
            batch_input_ids = batch[0].view(config["per_gpu_batch_size"]*2, -1)
            batch_attention_mask = batch[1].view(config["per_gpu_batch_size"]*2, -1)
            batch_token_type_ids = batch[2].view(config["per_gpu_batch_size"]*2, -1)
            batch_y_start = batch[3].view(config["per_gpu_batch_size"]*2)
            batch_y_end = batch[4].view(config["per_gpu_batch_size"]*2)
            batch_y = batch[5].view(config["per_gpu_batch_size"]*2)

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

            loss.backward()

            # todo: condition on the step in num of grad accum steps

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['max_grad_norm'],
            )
            optimizer.step()

            global_step += 1

            # todo: log training metrics
            # todo: eval

            output_dir = os.path.join(
                config['output_dir'],
                f'checkpoint-{global_step}'
            )
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir)

            #torch.save(config, os.path.join('training_args.bin'))
            torch.save(optimizer.state_dict(), optimizer_path)
            logger.info('Saving model checkpoint and'
                        f'optimizer state to {output_dir}')

            # todo: save model only if global_step % num_checkpts == 0 instead

            if 0 < config['max_steps'] < global_step:
                epoch_iterator.close()
                break
        if 0 < config['max_steps'] < global_step:
            train_iterator.close()
            break

    # todo: add SummaryWriter

    return global_step, tr_loss / global_step
