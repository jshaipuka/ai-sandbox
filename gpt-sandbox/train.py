import os

import torch
from torch import optim
from torch.nn import functional as F

from common import read_input, encode, get_batch, cwd, create_vocabulary
from model import BigramLanguageModel


def train():
    text = read_input()
    vocabulary = create_vocabulary(text)
    char_to_index = {c: i for i, c in enumerate(vocabulary)}
    data = torch.tensor(encode(char_to_index, text))
    training_data, validation_data = torch.split(data, int(0.9 * len(data)))

    model = BigramLanguageModel(len(vocabulary))

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(10000):
        x, y = get_batch(training_data, validation_data, split="training")
        logits = model(x)
        loss = F.cross_entropy(logits.permute(0, 2, 1), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(epoch, loss.item())

    file_name = "bigram_model.pt"
    torch.save(model.state_dict(), os.path.join(cwd, file_name))
    print("Model has been saved as", file_name)


if __name__ == "__main__":
    train()
