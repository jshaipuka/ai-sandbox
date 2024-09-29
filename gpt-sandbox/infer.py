import importlib
import os
import sys

import torch
import torch.nn.functional as F

from common import read_input, decode, cwd, create_vocabulary, device

OPTIONS = {
    "untrained_bigram_model": ("bigram_model", "BigramModel", "untrained_bigram_model.pt"),
    "bigram_model": ("bigram_model", "BigramModel", "bigram_model.pt"),
    "basic_gpt_model": ("basic_gpt_model", "GPT", "basic_gpt_model.pt"),
    "gpt_model": ("gpt_model", "GPT", "gpt_model.pt")
}


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

    (module_name, class_name, model_file_name) = OPTIONS[sys.argv[1] if len(sys.argv) >= 2 else "gpt_model"]
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    block_size = getattr(module, "BLOCK_SIZE")

    text = read_input()
    vocabulary = create_vocabulary(text)

    model = load_model(len(vocabulary), class_, model_file_name).to(device)
    indices = torch.zeros((1, 1), dtype=torch.long).to(device)
    prediction = generate(model, indices, vocabulary, block_size, max_new_tokens=10000)
    print(decode(vocabulary, prediction[0].tolist()))


if __name__ == "__main__":
    infer()
