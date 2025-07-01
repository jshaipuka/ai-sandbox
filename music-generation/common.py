import os
from enum import Enum

import regex as re
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax


class ModelType(Enum):
    GRU = 1,
    LSTM = 2


cwd = os.path.dirname(__file__)

BATCH_SIZE = 64
SEQ_LENGTH = 100
HIDDEN_DIM = 1024

# Change it to change the model.
DEFAULT_MODEL_TYPE = ModelType.LSTM


class GRUModel(nn.Module):

    def __init__(self, vocabulary_size, embedding_dim, hidden_dim):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1)
        self.linear = nn.Linear(hidden_dim, vocabulary_size)

    def forward(self, sequence, hidden):
        embedded = self.embedding(sequence)
        prediction, hidden = self.gru(embedded.transpose(1, 0), hidden)
        scores = self.linear(prediction.transpose(1, 0))
        return log_softmax(scores, dim=-1), hidden


class LSTMModel(nn.Module):

    def __init__(self, vocabulary_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocabulary_size)

    def forward(self, sequence, hidden):
        embedded = self.embedding(sequence)
        prediction, hidden = self.lstm(embedded.transpose(1, 0), hidden)
        scores = self.linear(prediction.transpose(1, 0))
        return log_softmax(scores, dim=-1), hidden


def determine_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def extract_song_snippet(text):
    pattern = "(^|\n\n)(.*?)\n\n"
    search_results = re.findall(pattern, text, overlapped=True, flags=re.DOTALL)
    songs = [song[1] for song in search_results]
    print("Found {} songs in text".format(len(songs)))
    return songs


def load_songs():
    with open(os.path.join(cwd, "data", "irish.abc"), "r", encoding="utf-8") as f:
        text = f.read()
    return extract_song_snippet(text)
