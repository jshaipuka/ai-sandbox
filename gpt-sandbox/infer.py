import os
import sys

import torch
import torch.nn.functional as F
import importlib

from common import read_input, decode, cwd, create_vocabulary, device


def load_model(vocabulary_size, model_class, model_file_name):
    model = model_class(vocabulary_size)
    model.load_state_dict(torch.load(os.path.join(cwd, "models", model_file_name), map_location=torch.device(device)))
    model.eval()
    return model


def generate(model, indices, vocabulary, block_size, max_new_tokens):
    prediction = torch.clone(indices)
    for _ in range(max_new_tokens):
        logits = model(prediction[:, -block_size:])
        last_timestamp = logits[:, -1, :]
        probability_distribution = F.softmax(last_timestamp, dim=-1)
        next_index = torch.multinomial(probability_distribution, num_samples=1)
        print(vocabulary[next_index], end="")
        sys.stdout.flush()
        prediction = torch.cat((prediction, next_index), dim=1)
    return prediction


def infer():
    print(f"Device is {device}")

    module = importlib.import_module(sys.argv[1])
    class_ = getattr(module, sys.argv[2])
    block_size = getattr(module, "BLOCK_SIZE")

    text = read_input()
    vocabulary = create_vocabulary(text)

    model = load_model(len(vocabulary), class_, sys.argv[3]).to(device)
    indices = torch.zeros((1, 1), dtype=torch.long).to(device)
    prediction = generate(model, indices, vocabulary, block_size, max_new_tokens=10000)
    print(decode(vocabulary, prediction[0].tolist()))


if __name__ == "__main__":
    infer()
