import os
from enum import Enum

import torch

cwd = os.path.dirname(__file__)

device = "cuda" if torch.cuda.is_available() else "cpu"


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


def get_batch(training_data, validation_data, batch_size, block_size, split: Split):
    data = training_data if split == Split.TRAINING else validation_data
    indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in indices])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in indices])
    return x, y
