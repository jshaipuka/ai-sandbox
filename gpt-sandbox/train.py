from typing import Literal

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

BATCH_SIZE = 32
BLOCK_SIZE = 8


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, indices, targets=None):
        # For each i in indices take ith row of the (vocab_size, vocab_size) table. The row will be the logits of the (i + 1)th character.
        # So, in short, each index i in indices is mapped to a row of numbers called logits.
        # In general, the length of the row will be not vocab_size, but whatever you passed as the 2nd parameter to nn.Embedding.
        # But in our case it's vocab_size.
        logits = self.token_embedding_table(indices)

        if targets is None:
            return logits, None
        else:
            b, t, c = logits.shape
            loss = F.cross_entropy(logits.view(b * t, c), targets.view(b * t))
            return logits, loss

    def infer(self, indices, max_new_tokens):
        prediction = torch.clone(indices)
        for _ in range(max_new_tokens):
            logits, _ = self(prediction)
            last_timestamp = logits[:, -1, :]
            p = F.softmax(last_timestamp, dim=-1)
            next_index = torch.multinomial(p, num_samples=1)
            prediction = torch.cat((prediction, next_index), dim=1)
        return prediction


def read_input():
    with open("input.txt", "r", encoding="utf-8") as f:
        return f.read()


def encode(char_to_index, string):
    return [char_to_index[c] for c in string]


def decode(index_to_char, indices):
    return "".join([index_to_char[i] for i in indices])


def get_batch(training_data, validation_data, split: Literal["training", "validation"]):
    data = training_data if split == "training" else validation_data
    indices = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    return torch.stack([data[i:i + BLOCK_SIZE] for i in indices]), torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in indices])


def train():
    text = read_input()
    vocabulary = sorted(set(text))
    char_to_index = {c: i for i, c in enumerate(vocabulary)}
    data = torch.tensor(encode(char_to_index, text))
    training_data, validation_data = torch.split(data, int(0.9 * len(data)))

    model = BigramLanguageModel(len(vocabulary))

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(10000):
        x, y = get_batch(training_data, validation_data, split="training")
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(epoch, loss.item())

    indices = torch.zeros((1, 1), dtype=torch.long)
    prediction = model.infer(indices, max_new_tokens=1000)
    print(decode(vocabulary, prediction[0].tolist()))


if __name__ == "__main__":
    train()
