# script that used the original dataset
# https://www.kaggle.com/c/tensorflow2-question-answering/data?select=simplified-nq-train.jsonl
# to drop all the samples that have at least 1 long answer (17G -> 8G)

from typing import Dict

import json


ORIG_DATA_PATH = 'data/simplified-nq-train.jsonl'
PREPROCESSED_DATA_PATH = 'data/preprocessed-nq-train.jsonl'


def has_answer(sample: Dict) -> bool:
    annotation = sample['annotations'][0]
    if (
            annotation['yes_no_answer'] != "NONE" or
            annotation['short_answers'] or
            annotation['long_answer']['candidate_index'] != -1
    ):
        return True
    return False


def has_la_answer(sample: Dict) -> bool:
    if len(sample['long_answer_candidates']) >= 1:
        return True
    return False


if __name__=='__main__':
    lines_count = 0
    lines_w_answer_count = 0
    data_iterator = open(ORIG_DATA_PATH)
    with open(PREPROCESSED_DATA_PATH, 'w') as fp:
        for data_line in data_iterator:
            sample = json.loads(data_line)
            lines_count += 1
            if lines_count % 100000 == 0:
                print(f'processed {lines_count} samples')
            if has_answer(sample):
                json.dump(sample, fp)
                fp.write('\n')
                lines_w_answer_count += 1

    print('Finished preprocessing the data',
          f'Saved {lines_w_answer_count} out of {lines_count} samples to {PREPROCESSED_DATA_PATH}'
          )