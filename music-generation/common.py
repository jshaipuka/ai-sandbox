import os

import regex as re
import torch.nn as nn
from torch.nn.functional import log_softmax

cwd = os.path.dirname(__file__)

BATCH_SIZE = 256
SEQ_LENGTH = 500
HIDDEN_DIM = 1024


class Model(nn.Module):

    def __init__(self, vocabulary_size, embedding_dim, hidden_dim):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocabulary_size)

    def forward(self, sequence, hidden):
        embedded = self.embedding(sequence)
        prediction, hidden = self.lstm(embedded, hidden)
        scores = self.linear(prediction)
        return log_softmax(scores, dim=-1), hidden


def extract_song_snippet(text):
    pattern = '(^|\n\n)(.*?)\n\n'
    search_results = re.findall(pattern, text, overlapped=True, flags=re.DOTALL)
    songs = [song[1] for song in search_results]
    print("Found {} songs in text".format(len(songs)))
    return songs


def load_songs():
    with open(os.path.join(cwd, "data", "irish.abc"), "r") as f:
        text = f.read()
    return extract_song_snippet(text)