import os

import torch
import torch.nn.functional as F

from common import read_input, decode, cwd, create_vocabulary, device, BLOCK_SIZE
from gpt_model import GPT


def load_model(vocabulary_size):
    model = GPT(vocabulary_size)
    model.load_state_dict(torch.load(os.path.join(cwd, "models", "gpt_model.pt")))
    model.eval()
    return model


def generate(model, indices, max_new_tokens):
    prediction = torch.clone(indices)
    for _ in range(max_new_tokens):
        logits = model(prediction[:, -BLOCK_SIZE:])
        last_timestamp = logits[:, -1, :]
        probability_distribution = F.softmax(last_timestamp, dim=-1)
        next_index = torch.multinomial(probability_distribution, num_samples=1)
        prediction = torch.cat((prediction, next_index), dim=1)
    return prediction


def infer():
    print(f"Device is {device}")
    text = read_input()
    vocabulary = create_vocabulary(text)
    model = load_model(len(vocabulary)).to(device)
    indices = torch.zeros((1, 1), dtype=torch.long).to(device)
    prediction = generate(model, indices, max_new_tokens=1000)
    print(decode(vocabulary, prediction[0].tolist()))


if __name__ == "__main__":
    infer()
