import os
from enum import Enum

import torch

cwd = os.path.dirname(__file__)

BATCH_SIZE = 32
BLOCK_SIZE = 8


class Split(Enum):
    TRAINING = 1
    VALIDATION = 2


def read_input():
    with open("input.txt", "r", encoding="utf-8") as f:
        return f.read()


def create_vocabulary(text):
    return sorted(set(text))


def encode(char_to_index, string):
    return [char_to_index[c] for c in string]


def decode(index_to_char, indices):
    return "".join([index_to_char[i] for i in indices])


def get_batch(training_data, validation_data, split: Split):
    data = training_data if split == Split.TRAINING else validation_data
    indices = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    return torch.stack([data[i:i + BLOCK_SIZE] for i in indices]), torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in indices])
