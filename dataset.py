from torch.utils.data import Dataset, IterableDataset
import torch
import json


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
    def __init__(self, filepath):
        self.filepath = filepath

    def get_example(self, line):
        example = json.loads(line)
        return example[1:]

    def __iter__(self):
        iterator = open(self.filepath)
        mapped_iterator = map(self.get_example, iterator)
        return mapped_iterator
