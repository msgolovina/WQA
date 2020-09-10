from torch.utils.data import Dataset
import torch
import json


class NQDataset(Dataset):
    def __init__(self, filepath, tokenizer):
        self.filepath = filepath
        self.tokenizer = tokenizer

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

