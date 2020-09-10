from typing import Dict, Tuple

from utils import random_sample

from torch.utils.data import IterableDataset
import numpy as np
import re
import json
import torch


class NQIterableDataset(IterableDataset):
    def __init__(self, filepath, tokenizer, max_seq_len, batch_size):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.sep_token_index = tokenizer.convert_tokens_to_ids('[SEP]')
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    @staticmethod
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

    def qp_to_tokens(self,
                     q_words,
                     is_pos,
                     p_words,
                     p_start,
                     p_end,
                     sa_start=None,
                     sa_end=None,
                     ):

        q_tokens = self.tokenizer.tokenize(q_words)
        max_answer_tokens = self.max_seq_len - len(q_tokens) - 3

        p_2_tokens_index = []
        p_tokens = []

        for j, word in enumerate(p_words):
            p_2_tokens_index.append(len(p_tokens))

            # skip paragraph tag tokens
            if re.match(r'<.+>', word):
                continue

            word_tokens = self.tokenizer.tokenize(word)
            if len(p_tokens) + len(word_tokens) > max_answer_tokens:
                break
            p_tokens += word_tokens

        qp_tokens = ['[CLS]'] + q_tokens + ['[SEP]'] + p_tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(qp_tokens)

        start = -1
        end = -1
        if is_pos:
            if sa_start is not None and sa_end is not None:
                if (sa_start >= p_start and sa_end <= p_end and sa_end - p_start < len(p_2_tokens_index)):
                    start = p_2_tokens_index[sa_start - p_start] + len(q_tokens) + 2
                    end = p_2_tokens_index[sa_end - p_start] + len(q_tokens) + 2

        return qp_tokens, input_ids, start, end

    def get_all_data(self, line):
        sample = json.loads(line)
        annotation = sample['annotations'][0]
        la_candidates = sample['long_answer_candidates']
        is_pos, class_label = self.get_class_label(annotation)

        sample_id = sample['example_id']
        doc_words = sample['document_text'].split()

        # positive candidate
        pos_index = annotation['long_answer']['candidate_index']
        pos_candidate = la_candidates[pos_index]
        pos_start = pos_candidate['start_token']
        pos_end = pos_candidate['end_token']
        pos_words = doc_words[pos_start:pos_end]

        # random negative candidate
        distribution = np.ones((len(la_candidates),), dtype=np.float32)
        if is_pos:
            distribution[pos_index] = 0.
        distribution /= len(distribution)
        neg_index = random_sample(distribution)
        neg_candidate = la_candidates[neg_index]
        neg_start = neg_candidate['start_token']
        neg_end = neg_candidate['end_token']
        neg_words = doc_words[neg_start:neg_end]
        question_words = sample['question_text']

        short_answers = annotation['short_answers']

        if short_answers:
            sa_start = short_answers[0]['start_token']
            sa_end = short_answers[0]['end_token']
        else:
            sa_start, sa_end = None, None

        # pos words to tokens
        pos_input_tokens, pos_input_ids, pos_start, pos_end = self.qp_to_tokens(
            question_words, True, pos_words, pos_start, pos_end, sa_start, sa_end)

        # neg words to tokens
        neg_input_tokens, neg_input_ids, neg_start, neg_end = self.qp_to_tokens(
            question_words, False, neg_words, neg_start, neg_end)

        # initialize input arrays
        batch_ids = [line]
        batch_size = 2 * len(batch_ids)
        input_ids = np.zeros((batch_size, self.max_seq_len), dtype=np.int64)
        token_type_ids = np.ones((batch_size, self.max_seq_len), dtype=np.int64)
        y_start = np.zeros((batch_size,), dtype=np.int64)
        y_end = np.zeros((batch_size,), dtype=np.int64)
        y = np.zeros((batch_size,), dtype=np.int64)

        # fill in the arrays
        i = 0
        y[2 * i] = class_label
        y[2 * i + 1] = 0
        input_ids[2 * i, :len(pos_input_ids)] = pos_input_ids
        input_ids[2 * i + 1, :len(neg_input_ids)] = neg_input_ids

        token_type_ids[2 * i, :len(pos_input_ids)] = [0 if j <=
                                                                pos_input_ids.index(
                                                                    self.sep_token_index) else 1 for j
                                                           in range(len(pos_input_ids))]
        token_type_ids[2 * i + 1, :len(neg_input_ids)] = [0 if j <=
                                                                    neg_input_ids.index(
                                                                        self.sep_token_index) else 1 for j
                                                               in range(len(neg_input_ids))]
        y_start[2 * i] = pos_start
        y_start[2 * i + 1] = neg_start
        y_end[2 * i] = pos_end
        y_end[2 * i + 1] = neg_end
        attention_mask = input_ids > 0

        return torch.from_numpy(input_ids), \
               torch.from_numpy(attention_mask), \
               torch.from_numpy(token_type_ids), \
               torch.LongTensor(y_start), \
               torch.LongTensor(y_end), \
               torch.LongTensor(y)


    def __iter__(self):
        batch_size = self.batch_size
        iterator = open(self.filepath)
        mapped_iterator = map(self.get_all_data, iterator)
        # todo: is it less efficient than DataLoader since it does not parallelize stuff?
        batch = []
        drop_last = True

        while True:
            next_iter = next(mapped_iterator)
            batch.append(next_iter)
            while len(batch) >= batch_size:
                if len(batch) == batch_size:
                    batch = [
                        torch.stack([i[j] for i in batch]).view(batch_size * 2, -1).squeeze() for j in
                        range(len(batch[0]))
                    ]
                    yield batch
                    batch = []
                else:
                    return_batch = batch[:batch_size]
                    batch = batch[batch_size:]
                    yield return_batch
            if len(batch) > 0 and not drop_last:
                batch = [
                    torch.stack([i[j] for i in batch]).view(batch_size * 2, -1).squeeze() for j in
                    range(len(batch[0]))
                ]
                yield batch

