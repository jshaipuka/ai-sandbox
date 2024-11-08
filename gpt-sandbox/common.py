import os
from enum import Enum

import torch

cwd = os.path.dirname(__file__)


OPTIONS = {
    "untrained_bigram_model": ("bigram_model", "BigramModel", "untrained_bigram_model.pt"),
    "bigram_model": ("bigram_model", "BigramModel", "bigram_model.pt"),
    "basic_gpt_model": ("basic_gpt_model", "GPT", "basic_gpt_model.pt"),
    "gpt_model": ("gpt_model", "GPT", "gpt_model.pt")
}


def determine_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


device = determine_device()


class Split(Enum):
    TRAINING = 1
    VALIDATION = 2


def read_input():
    with open('input.txt', 'r', encoding='utf-8') as f:
        return f.read()


def create_vocabulary(text):
    return sorted(set(text))


def encode(char_to_index, string):
    return [char_to_index[c] for c in string]


def decode(index_to_char, indices):
    return ''.join([index_to_char[i] for i in indices])


def get_batch(training_data, validation_data, batch_size, block_size, split: Split):
    data = training_data if split == Split.TRAINING else validation_data
    indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in indices])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in indices])
    return x, y
