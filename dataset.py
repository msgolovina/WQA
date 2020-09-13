from torch.utils.data import Dataset, IterableDataset
import torch
import json
import numpy as np


class NQDataset(Dataset):
    def __init__(self, filepath, tokenizer):
        self.filepath = filepath
        self.tokenizer = tokenizer # todo: ??

        data_iter = open(filepath)
        self.dataset = []
        for line in data_iter:
            self.dataset.append(json.loads(line))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item_lists = self.dataset[index]
        item_tensors = [torch.tensor(x) for x in item_lists]
        return item_tensors


class NQIterableTestDataset(IterableDataset):
    def __init__(self, filepath): #tokenizer
        self.filepath = filepath
        #self.tokenizer = tokenizer

    def get_example(self, line):
        example = json.loads(line)
        # question = example[0][0]
        # answer_idx = np.argmax(example[-1])
        # answer_label = np.max(example[-1])
        # input_ids, attention_mask, token_type_ids, start, end, y
        return example[1:]

    def __iter__(self):
        iterator = open(self.filepath)
        mapped_iterator = map(self.get_example, iterator)
        return mapped_iterator
