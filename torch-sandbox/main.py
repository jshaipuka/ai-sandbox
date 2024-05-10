import os

import keras_core as keras
import numpy as np
import regex as re
import torch
from torch.nn.functional import softmax
from tqdm import tqdm

cwd = os.path.dirname(__file__)

BATCH_SIZE = 64


def assert_gpu_available():
    assert torch.cuda.is_available(), "GPU is not available"


def extract_song_snippet(text):
    pattern = '(^|\n\n)(.*?)\n\n'
    search_results = re.findall(pattern, text, overlapped=True, flags=re.DOTALL)
    songs = [song[1] for song in search_results]
    print("Found {} songs in text".format(len(songs)))
    return songs


def load_songs():
    with open(os.path.join(cwd, "data", "irish.abc"), "r") as f:
        text = f.read()
    return extract_song_snippet(text)


def vectorize_string(string, character_to_index):
    return np.array([character_to_index[character] for character in string])


def get_batch(vectorized_songs, seq_length, batch_size):
    random_indexes = np.random.choice(vectorized_songs.shape[0] - seq_length - 1, batch_size)
    x_batch = np.reshape([vectorized_songs[i:i + seq_length] for i in random_indexes], [batch_size, seq_length])
    y_batch = np.reshape([vectorized_songs[i + 1:i + seq_length + 1] for i in random_indexes], [batch_size, seq_length])
    return x_batch, y_batch


def build_model(vocabulary_size, embedding_dim, rnn_units, batch_size):
    shape = (100,)  # TODO: check the shape parameter docs, it was (None, ) previously
    stateful = False  # TODO: check why setting it to True fails the call to loss.backward()
    return keras.Sequential([
        keras.layers.Input(shape=shape, batch_size=batch_size),
        keras.layers.Embedding(vocabulary_size, embedding_dim),
        keras.layers.LSTM(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform', recurrent_activation='sigmoid', stateful=stateful),
        keras.layers.Dense(vocabulary_size)
    ])


def generate_text(model, char_to_index, index_to_char, start_string, generation_length=1000):
    input_eval = [char_to_index[s] for s in start_string]
    input_eval = torch.unsqueeze(torch.tensor(input_eval), 0)

    text_generated = []

    tqdm._instances.clear()

    for i in tqdm(range(generation_length)):
        predictions = torch.squeeze(model(input_eval), 0)
        predicted_index = torch.multinomial(softmax(predictions, dim=0), 1, replacement=True)[-1, 0]
        input_eval = torch.unsqueeze(torch.unsqueeze(predicted_index, 0), 0)
        text_generated.append(index_to_char[predicted_index.item()])

    return start_string + ''.join(text_generated)


def load_model(vocabulary_size, file_name):
    model = build_model(vocabulary_size, embedding_dim=256, rnn_units=1024, batch_size=1)
    model.load_state_dict(torch.load(os.path.join(cwd, "models", file_name)))
    model.eval()
    return model


def main():
    songs = load_songs()
    songs_joined = "\n\n".join(songs)
    vocabulary = sorted(set(songs_joined))
    char_to_index = {u: i for i, u in enumerate(vocabulary)}
    index_to_char = np.array(vocabulary)

    vectorized_songs = vectorize_string(songs_joined, char_to_index)

    model = build_model(len(vocabulary), embedding_dim=256, rnn_units=1024, batch_size=BATCH_SIZE)
    model.summary()

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    for i in range(800):
        input_batch, target_batch = get_batch(vectorized_songs, seq_length=100, batch_size=BATCH_SIZE)

        prediction = model(input_batch)
        loss = loss_fn(prediction.permute((0, 2, 1)).cpu(), torch.from_numpy(target_batch).long())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            print(i, loss.item())
        if i and i % 100 == 0:
            torch.save(model.state_dict(), os.path.join(cwd, "models", "model_" + str(i) + ".pt"))
            print("Model has been saved")

    print(generate_text(model, char_to_index, index_to_char, 'X'))


if __name__ == '__main__':
    main()
