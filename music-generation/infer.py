import os
import tempfile

import numpy as np
import torch

from common import load_songs, load_model, extract_song_snippet, HIDDEN_DIM

cwd = os.path.dirname(__file__)


def generate_text(model, char_to_index, index_to_char, start_string, generation_length=1000):
    h_0, c_0 = torch.zeros(1, 1, HIDDEN_DIM), torch.zeros(1, 1, HIDDEN_DIM)
    hidden = (h_0, c_0)
    input_eval = [char_to_index[s] for s in start_string]

    text_generated = []

    for i in range(generation_length):
        predictions, hidden = model(torch.unsqueeze(torch.tensor(input_eval), 0), hidden)
        predicted_index = torch.multinomial(predictions.view(-1).exp(), 1).item()
        input_eval = [predicted_index]
        text_generated.append(index_to_char[predicted_index])
        if (i + 1) % 10 == 0:
            print("Predicted", i + 1, "characters out of", generation_length)

    return start_string + ''.join(text_generated)


def save_song_to_abc(song, file_name):
    with open(file_name, "w") as f:
        f.write(song)


def infer():
    songs = load_songs()
    songs_joined = "\n\n".join(songs)
    vocabulary = sorted(set(songs_joined))
    char_to_index = {u: i for i, u in enumerate(vocabulary)}
    index_to_char = np.array(vocabulary)

    trained_model = load_model(len(vocabulary), "main_model_1599.pt")
    predicted_text = generate_text(trained_model, char_to_index, index_to_char, "X", 2000)
    print(predicted_text)
    generated_songs = extract_song_snippet(predicted_text)
    temp_dir_name = tempfile.mkdtemp()
    for i, song in enumerate(generated_songs):
        abc_file = os.path.join(temp_dir_name, "song_{}.abc".format(i))
        save_song_to_abc(song, abc_file)
        mid_file = os.path.join(temp_dir_name, "song_{}.mid".format(i))
        cmd = "{} {} -o {}".format("abc2midi", abc_file, mid_file)
        os.system(cmd)
    print("Saved", len(generated_songs), "songs to", temp_dir_name)


infer()
