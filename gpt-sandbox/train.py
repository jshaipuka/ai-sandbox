import importlib
import os
import sys
import time

import torch
from torch import optim
from torch.nn import functional as F

from common import read_input, encode, get_batch, cwd, create_vocabulary, Split, device, OPTIONS

EVAL_INTERVAL = 500
EVAL_ITERS = 200


@torch.no_grad()
def estimate_loss(model, training_data, validation_data, batch_size, block_size):
    out = {}
    model.eval()
    for split in list(Split):
        losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            x, y = get_batch(training_data, validation_data, batch_size, block_size, split)
            logits = model(x.to(device))
            loss = F.cross_entropy(logits.permute(0, 2, 1), y.to(device))
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train():
    print(f"Device is {device}")

    torch.set_float32_matmul_precision('high')

    (module_name, class_name, model_file_name) = OPTIONS[sys.argv[1] if len(sys.argv) >= 2 else "gpt_model"]
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    block_size = getattr(module, "BLOCK_SIZE")
    batch_size = getattr(module, "BATCH_SIZE")
    learning_rate = getattr(module, "LEARNING_RATE")
    num_epochs = getattr(module, "NUM_EPOCHS")

    text = read_input()
    vocabulary = create_vocabulary(text)
    char_to_index = {c: i for i, c in enumerate(vocabulary)}
    data = torch.tensor(encode(char_to_index, text))
    training_data, validation_data = torch.split(data, int(0.9 * len(data)))

    model = class_(len(vocabulary)).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    dt_ms = 0
    for epoch in range(num_epochs):
        if epoch % EVAL_INTERVAL == 0 or epoch == num_epochs - 1:
            losses = estimate_loss(model, training_data, validation_data, batch_size, block_size)
            print(f"Step: {epoch} took time: {dt_ms:.2f} ms: training loss {losses[Split.TRAINING]:.4f}, validation loss {losses[Split.VALIDATION]:.4f}")

        t0 = time.time()
        x, y = get_batch(training_data, validation_data, batch_size, block_size, split=Split.TRAINING)
        logits = model(x.to(device))
        loss = F.cross_entropy(logits.permute(0, 2, 1), y.to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        t1 = time.time()
        dt_ms = (t1 - t0) * 1000

    torch.save(model.state_dict(), os.path.join(cwd, "models", model_file_name))
    print(f"Model has been saved as {model_file_name}")


if __name__ == "__main__":
    train()
