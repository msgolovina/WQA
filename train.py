from dataset import NQIterableDataset
from train_function import train
from models import BertForQA

import logging
import os
from transformers import BertTokenizer, BertConfig, AdamW

logger = logging.getLogger(__name__)

TRAIN_ARGS = {} # todo: create config for global vars
TRAIN_PATH = 'data/simplified-nq-train.jsonl'
BERT_PATH = 'bert-base-uncased'
MAX_SEQ_LEN = 384
NUM_LABELS = 5
DEVICE = 'cuda'
OUTPUT_DIR = 'trained_models/'

if __name__ == '__main__':

    tokenizer = BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
    bert_config = BertConfig.from_pretrained(BERT_PATH)
    bert_config.num_labels = NUM_LABELS
    model = BertForQA.from_pretrained(BERT_PATH, config=bert_config)
    model.to(DEVICE)

    logger.info(f'Training/eval params: {TRAIN_ARGS}')

    if DO_TRAIN:
        dataset = NQIterableDataset(TRAIN_PATH, tokenizer, MAX_SEQ_LEN)

        global_step, tr_loss = train(
            dataset, model, TRAIN_ARGS
        )

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        model = BertForQA.from_pretrained(OUTPUT_DIR)
        model.to(DEVICE)

        # todo: loading from checkpt
        # todo: add eval
        # todo: add metrics






