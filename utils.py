from typing import Dict, Tuple

import numpy as np
import torch
import re
import random


def random_sample(distribution):
    temp = np.random.random()
    value = 0.
    idx = 0
    for index in range(len(distribution)):
        idx = index
        value += distribution[index]
        if value > temp:
            break
    return idx  # todo: or is it idx?


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def get_class_label(annotation: Dict) -> Tuple[bool, int]:
    """
    classes:
    0: no answer
    1: long answer only
    2: short answer and long answer
    3: no
    4: yes
    """
    is_pos = True
    if annotation['yes_no_answer'] == 'YES':
        class_label = 4
    elif annotation['yes_no_answer'] == 'NO':
        class_label = 3
    elif annotation['short_answers']:
        class_label = 2
    elif annotation['long_answer']['candidate_index'] != -1:
        class_label = 1
    else:
        class_label = 0
        is_pos = False

    return is_pos, class_label


def qp_to_tokens(
                 q_words,
                 is_pos,
                 p_words,
                 p_start,
                 p_end,
                 tokenizer,
                 max_seq_len,
                 sa_start=None,
                 sa_end=None,
                 ):

    q_tokens = tokenizer.tokenize(q_words)
    max_answer_tokens = max_seq_len - len(q_tokens) - 3

    p_2_tokens_index = []
    p_tokens = []

    for j, word in enumerate(p_words):
        p_2_tokens_index.append(len(p_tokens))

        # skip paragraph tag tokens
        if re.match(r'<.+>', word):
            continue

        word_tokens = tokenizer.tokenize(word)
        if len(p_tokens) + len(word_tokens) > max_answer_tokens:
            break
        p_tokens += word_tokens

    qp_tokens = ['[CLS]'] + q_tokens + ['[SEP]'] + p_tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(qp_tokens)

    start = -1
    end = -1
    if is_pos:
        if sa_start is not None and sa_end is not None:
            if sa_start >= p_start and sa_end <= p_end and sa_end - p_start < len(p_2_tokens_index):
                start = p_2_tokens_index[sa_start - p_start] + len(q_tokens) + 2
                end = p_2_tokens_index[sa_end - p_start] + len(q_tokens) + 2

    return qp_tokens, input_ids, start, end