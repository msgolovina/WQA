from utils import qp_to_tokens, get_class_label

import numpy as np
import json
import yaml
from transformers import BertTokenizer

CONFIG_PATH = 'logs/test_run/config.yml'
TEST_PATH = 'data/preprocessed_test.jsonl'
FINAL_TEST_PATH = 'data/test.jsonl'

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
tokenizer = BertTokenizer.from_pretrained(
        config['pretrained_bert_name'],
        do_lower_case=True
    )
sep_token_index = tokenizer.convert_tokens_to_ids('[SEP]')
max_seq_len = config['max_seq_len']


def get_test_data(line):
    '''
    for test data we need to store all the paragraphs encoded so that we
    could pass them to the model as well as annotation
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
    return question_words, la_candidates, sa_start, sa_end, input_ids, attention_mask, token_type_ids, y_start, y_end, y

if __name__=='__main__':
    lines_count = 0
    data_iterator = open(TEST_PATH)
    with open(FINAL_TEST_PATH, 'w') as fp:
        for data_line in data_iterator:
            lines_count += 1
            if lines_count % 500 == 0:
                print(f'processed {lines_count} lines')
            preprocessed_line = get_test_data(data_line)
            preprocessed_line = [preprocessed_line[:4]]+[x.tolist() for x in preprocessed_line[4:]]
            json.dump(preprocessed_line, fp)
            fp.write('\n')
