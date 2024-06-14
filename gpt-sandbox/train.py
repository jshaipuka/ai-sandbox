import os

import torch
from torch import optim
from torch.nn import functional as F

from common import read_input, encode, get_batch, cwd, create_vocabulary, Split
from model import BigramLanguageModel

EVAL_INTERVAL = 300
EVAL_ITERS = 200


@torch.no_grad()
def estimate_loss(model, device, training_data, validation_data):
    out = {}
    model.eval()
    for split in list(Split):
        losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            x, y = get_batch(training_data, validation_data, split)
            logits = model(x.to(device))
            loss = F.cross_entropy(logits.permute(0, 2, 1), y.to(device))
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = read_input()
    vocabulary = create_vocabulary(text)
    char_to_index = {c: i for i, c in enumerate(vocabulary)}
    data = torch.tensor(encode(char_to_index, text))
    training_data, validation_data = torch.split(data, int(0.9 * len(data)))

    model = BigramLanguageModel(len(vocabulary)).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(10000):
        if epoch % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, device, training_data, validation_data)
            print(f"Step: {epoch}: training loss {losses[Split.TRAINING]:.4f}, validation loss {losses[Split.VALIDATION]:.4f}")

        x, y = get_batch(training_data, validation_data, split=Split.TRAINING)
        logits = model(x.to(device))
        loss = F.cross_entropy(logits.permute(0, 2, 1), y.to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    file_name = "bigram_model.pt"
    torch.save(model.state_dict(), os.path.join(cwd, file_name))
    print(f"Model has been saved as {file_name}")


if __name__ == "__main__":
    train()
