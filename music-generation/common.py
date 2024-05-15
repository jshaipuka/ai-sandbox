import os

import regex as re
import torch
from torch.nn.functional import log_softmax

cwd = os.path.dirname(__file__)

BATCH_SIZE = 64
HIDDEN_DIM = 1024


class Model(torch.nn.Module):

    def __init__(self, vocabulary_size, embedding_dim, hidden_dim):
        super(Model, self).__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = torch.nn.Linear(hidden_dim, vocabulary_size)

    def forward(self, sequence, hidden):
        embedded = self.embedding(sequence)  # (64, 100, 256)
        prediction, hidden = self.lstm(embedded, hidden)  # (64, 100, 1024)
        scores = self.linear(prediction)  # (64, 100, 83)
        return log_softmax(scores, dim=-1), hidden  # (64, 100, 83)


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


def load_model(vocabulary_size, file_name):
    model = Model(vocabulary_size, 256, HIDDEN_DIM)
    model.load_state_dict(torch.load(os.path.join(cwd, "models", file_name)))
    model.eval()
    return model
