import os
from enum import Enum

import torch

cwd = os.path.dirname(__file__)

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
BLOCK_SIZE = 256
EMBEDDING_DIM = 384
NUM_HEADS = 6
NUM_LAYERS = 6
LEARNING_RATE = 3e-4
NUM_EPOCHS = 5000
DROPOUT = 0.2


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
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in indices])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in indices])
    return x, y
