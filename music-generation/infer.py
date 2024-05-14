import os

import numpy as np
import torch
from torch.nn.functional import softmax

from common import load_songs, load_model

cwd = os.path.dirname(__file__)


def generate_text(model, char_to_index, index_to_char, start_string, generation_length=1000):
    c_0, h_0 = torch.zeros(1, 1, 1024), torch.zeros(1, 1, 1024)
    hidden = (h_0, c_0)
    input_eval = [char_to_index[s] for s in start_string]

    text_generated = []

    for i in range(generation_length):
        predictions, hidden = model(sequence=torch.tensor(input_eval), hidden=hidden)
        predicted_index = torch.multinomial(softmax(torch.squeeze(predictions, 0), dim=0), 1)[-1].item()
        input_eval = [predicted_index]
        text_generated.append(index_to_char[predicted_index])
        if i % 10 == 0:
            print("Predicted character", i)

    return start_string + ''.join(text_generated)


def infer():
    songs = load_songs()
    songs_joined = "\n\n".join(songs)
    vocabulary = sorted(set(songs_joined))
    char_to_index = {u: i for i, u in enumerate(vocabulary)}
    index_to_char = np.array(vocabulary)

    trained_model = load_model(len(vocabulary), "main_model_700_pure_pytorch.pt")
    print(generate_text(trained_model, char_to_index, index_to_char, "X", 2000))


infer()
