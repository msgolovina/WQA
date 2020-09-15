from utils import qp_to_tokens, get_class_label
from models import BertForQA

import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    "--local-rank",
    type=int,
    default=-1,
    help="local_rank for distributed training on gpus",
)
parser.add_argument(
    "--model-checkpoint-path",
    type=str,
    default='logs/test_run/output/checkpoint-17000'
)

CONFIG_PATH = 'logs/test_run/config.yml'
TRAIN_PATH = 'data/preprocessed_train.jsonl'
FINAL_TRAIN_PATH = 'data/hnm_train.jsonl'

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
tokenizer = BertTokenizer.from_pretrained(
        config['pretrained_bert_name'],
        do_lower_case=True
    )
sep_token_index = tokenizer.convert_tokens_to_ids('[SEP]')
max_seq_len = config['max_seq_len']


def get_hnm_train_data(line):
    '''
    we need to get all the paragraphs encoded so that we
    could pass them to the model
    '''
    sample = json.loads(line)
    annotation = sample['annotations'][0]
    la_candidates = sample['long_answer_candidates']
    is_pos, class_label = get_class_label(annotation)

    sample_id = sample['example_id']
    doc_words = sample['document_text'].split()
    question_words = sample['question_text']
    # positive candidate index
    pos_index = annotation['long_answer']['candidate_index']
    short_answers = annotation['short_answers']
    if short_answers:
        sa_start = short_answers[0]['start_token']
        sa_end = short_answers[0]['end_token']
    else:
        sa_start, sa_end = None, None

    # initialize input arrays
    size = len(la_candidates)
    input_ids = np.zeros((size, max_seq_len), dtype=np.int64)
    token_type_ids = np.ones((size, max_seq_len), dtype=np.int64)
    y_start = np.zeros((size,), dtype=np.int64)
    y_end = np.zeros((size,), dtype=np.int64)
    y = np.zeros((size,), dtype=np.int64)

    for index, candidate in enumerate(la_candidates):
        start = candidate['start_token']
        end = candidate['end_token']
        words = doc_words[start:end]
        if index == pos_index:
            is_pos = True
        else:
            is_pos = False
        tokens, ids, start, end = qp_to_tokens(
            question_words, is_pos, words, start, end, tokenizer, max_seq_len, sa_start, sa_end)

        # fill in the arrays
        if is_pos:
            y[index] = class_label
        input_ids[index, :len(ids)] = ids
        token_type_ids[index, :len(ids)] = [0 if j <=
                        ids.index(sep_token_index) else 1 for j
                        in range(len(ids))]

        y_start[index] = start
        y_end[index] = end
    attention_mask = input_ids > 0
    return input_ids, attention_mask, token_type_ids, y_start, y_end, y


def get_hnm_samples(line, model):
    example = get_hnm_train_data(line)
    class_label = np.max(example[-1])
    answer = np.argmax(example[-1])
    example = [torch.tensor(t) for t in example]
    example = [t.to(args.device) for t in example]
    la_scores = []

    for idx in range(len(example[-1])):
        one_p = [x[idx] for x in example]
        input_ids = one_p[0].view(1, -1)
        attention_mask = one_p[1].view(1, -1)
        token_type_ids = one_p[2].view(1, -1)

        logits_start, logits_end, logits_class = model(
            input_ids, attention_mask, token_type_ids
        )
        no_answer_score = logits_class[0][0]
        la_scores.append(1 - no_answer_score.item())

    try:
        top3 = np.argpartition([-x for x in la_scores], 3)[:3]
        top2_neg = [x for x in top3 if x != answer][:2]
        idx_to_save = [answer, top2_neg[0], top2_neg[1]]
        samples_to_save = []
        for i in idx_to_save:
            input_ids = example[i][0].view(1, -1)
            attention_mask = example[i][1].view(1, -1)
            token_type_ids = example[i][2].view(1, -1)
            y_start = example[i][3]
            y_end = example[i][4]
            y = example[i][5]
            samples_to_save.append([input_ids, attention_mask, token_type_ids, y_start, y_end, y])
        return samples_to_save
    except:
        return None # todo: preprocess the data that had < 3 la candidates explicitly?


if __name__=='__main__':

    args = parser.parse_args()

    args.device = torch.device('cuda', args.local_rank)
    args.n_gpu = 1

    model = BertForQA.from_pretrained(args.model_checkpoint_path)
    model.to(args.device)

    data_iterator = open(TRAIN_PATH)
    with open(FINAL_TRAIN_PATH, 'w') as fp:
        with torch.no_grad():
            for data_line in tqdm(data_iterator):
                samples = get_hnm_samples(data_line, model)
                if samples is not None:
                    for sample in samples:
                        sample = [x.tolist() for x in sample]
                        json.dump(sample, fp)
                        fp.write('\n')
