import numpy as np
import torch
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