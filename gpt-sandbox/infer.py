import os

import torch

from common import read_input, decode, cwd, create_vocabulary, EMBEDDING_DIM
from model import GPT


def load_model(vocabulary_size, embedding_dim):
    model = GPT(vocabulary_size, embedding_dim)
    model.load_state_dict(torch.load(os.path.join(cwd, "gpt_model.pt")))
    model.eval()
    return model


def infer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device is {device}")
    text = read_input()
    vocabulary = create_vocabulary(text)
    model = load_model(len(vocabulary), EMBEDDING_DIM).to(device)
    indices = torch.zeros((1, 1), dtype=torch.long).to(device)
    prediction = model.infer(indices, max_new_tokens=1000)
    print(decode(vocabulary, prediction[0].tolist()))


if __name__ == "__main__":
    infer()
