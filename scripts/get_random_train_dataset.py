from utils import random_sample, qp_to_tokens, get_class_label

import numpy as np
import json
import yaml
from transformers import BertTokenizer

CONFIG_PATH = 'logs/test_run/config.yml'
TRAIN_PATH = 'data/preprocessed_train.jsonl'
FINAL_TRAIN_PATH = 'data/random_train_1.jsonl'

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
tokenizer = BertTokenizer.from_pretrained(
        config['pretrained_bert_name'],
        do_lower_case=True
    )
sep_token_index = tokenizer.convert_tokens_to_ids('[SEP]')
max_seq_len = config['max_seq_len']


def get_train_data(line):
    '''
    we don't need original question, article and paragraphs for the train dataset
    '''
    sample = json.loads(line)
    annotation = sample['annotations'][0]
    la_candidates = sample['long_answer_candidates']
    is_pos, class_label = get_class_label(annotation)

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
    pos_input_tokens, pos_input_ids, pos_start, pos_end = qp_to_tokens(
        question_words, True, pos_words, pos_start, pos_end, tokenizer, max_seq_len, sa_start, sa_end)

    # neg words to tokens
    neg_input_tokens, neg_input_ids, neg_start, neg_end = qp_to_tokens(
        question_words, False, neg_words, neg_start, neg_end, tokenizer, max_seq_len)

    # initialize input arrays
    batch_ids = [line]
    batch_size = 2 * len(batch_ids)
    input_ids = np.zeros((batch_size, max_seq_len), dtype=np.int64)
    token_type_ids = np.ones((batch_size, max_seq_len), dtype=np.int64)
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
                                                                sep_token_index) else 1 for j
                                                       in range(len(pos_input_ids))]
    token_type_ids[2 * i + 1, :len(neg_input_ids)] = [0 if j <=
                                                                neg_input_ids.index(
                                                                    sep_token_index) else 1 for j
                                                           in range(len(neg_input_ids))]
    y_start[2 * i] = pos_start
    y_start[2 * i + 1] = neg_start
    y_end[2 * i] = pos_end
    y_end[2 * i + 1] = neg_end
    attention_mask = input_ids > 0

    return input_ids, attention_mask, token_type_ids, y_start, y_end, y

    # return torch.from_numpy(input_ids), \
    #        torch.from_numpy(attention_mask), \
    #        torch.from_numpy(token_type_ids), \
    #        torch.LongTensor(y_start), \
    #        torch.LongTensor(y_end), \
    #        torch.LongTensor(y)


if __name__=='__main__':
    lines_count = 0
    data_iterator = open(TRAIN_PATH)
    with open(FINAL_TRAIN_PATH, 'w') as fp:
        for data_line in data_iterator:
            lines_count += 1
            if lines_count % 10000 == 0:
                print(f'processed {lines_count} lines')
            preprocessed_line = get_train_data(data_line)
            preprocessed_line = [x.tolist() for x in preprocessed_line]
            positive = [x[0] for x in preprocessed_line]
            negative = [x[1] for x in preprocessed_line]
            json.dump(positive, fp)
            fp.write('\n')
            json.dump(negative, fp)
            fp.write('\n')
