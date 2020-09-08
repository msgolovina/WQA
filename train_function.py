from utils import set_seed

from tqdm import tqdm, trange
from transformers import AdamW
from torch.nn import CrossEntropyLoss
import os
import torch
import logging


logger = logging.getLogger(__name__)


def triple_loss(preds, labels):
    start_preds, end_preds, class_preds = preds
    start_labels, end_labels, class_labels = labels

    start_loss = CrossEntropyLoss(ignore_index=-1)(start_preds, start_labels)
    end_loss = CrossEntropyLoss(ignore_index=-1)(end_preds, end_labels)
    class_loss = CrossEntropyLoss(ignore_index=-1)(class_preds, class_labels)

    return start_loss + end_loss + class_loss


def train(data_iterator, model, args):

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
            'weight_decay': args.weight_decay,
            'params': [
                param for name, param in model.named_parameters()
                if not any(nd_name in name for nd_name in no_decay)
            ]
        }
    ]

    optimizer = AdamW(
        optimizer_params,
        lr=args.learning_rate,
        eps=args.adam_epsilon
    )
    optimizer_path = os.path.join(args.model_name_or_path, 'optimizer.pt')
    if os.path.isfile(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path))
    model = ...

    # training
    logger.info('==========================')
    logger.info('---------TRAINING---------')
    logger.info('==========================')
    logger.info(f' Num epochs: {args.num_train_epochs}')
    logger.info(f' Batch size per GPU: {args.per_gpu_train_batch_size}')

    # todo: upload checkpoint if exists

    global_step = 1
    epochs_trained = 0
    tr_loss, logging_loss = 0., 0.

    model.zero_grad()

    train_iterator = trange(epochs_trained,
                            args.num_train_epochs,
                            desc='Epoch',
    )

    set_seed(args.seed, args.n_gpu)

    for iteration in train_iterator:

        epoch_iterator = tqdm(
            data_iterator,
            desc='Iteration',
        )

        for step, batch in enumerate(epoch_iterator):

            # todo: skip past trained steps if resuming training

            model.train()

            batch = [t.to(args.device) for t in batch]
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
            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()

            # todo: condition on the step in num of grad accum steps

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )
            optimizer.step()

            global_step += 1

            # todo: log training metrics
            # todo: eval

            output_dir = os.path.join(
                args.output_dir,
                f'checkpoint-{global_step}'
            )
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir)

            torch.save(args, os.path.join('training_args.bin'))
            torch.save(optimizer.state_dict(), optimizer_path)
            logger.info('Saving model checkpoint and'
                        f'optimizer state to {output_dir}')

            # todo: save model only if global_step % num_checkpts == 0 instead

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_step:
            train_iterator.close()
            break

    # todo: add SummaryWriter

    return global_step, tr_loss / global_step
