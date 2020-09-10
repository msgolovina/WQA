import random
import json

DATA_PATH = 'data/preprocessed-nq-train.jsonl'
TRAIN_PATH = 'data/preprocessed_train.jsonl'
TEST_PATH = 'data/preprocessed_test.jsonl'
TEST_SIZE = 0.1


if __name__=='__main__':
    lines_count = 0
    train_count = 0
    test_count = 0
    data_iterator = open(DATA_PATH)
    with open(TEST_PATH, 'w') as test_fp:
        with open(TRAIN_PATH, 'w') as train_fp:
            for data_line in data_iterator:
                lines_count += 1
                line = json.loads(data_line)
                to_train = True if random.random() > TEST_SIZE else False
                if to_train:
                    train_count += 1
                    json.dump(line, train_fp)
                    train_fp.write('\n')
                else:
                    test_count += 1
                    json.dump(line, test_fp)
                    test_fp.write('\n')