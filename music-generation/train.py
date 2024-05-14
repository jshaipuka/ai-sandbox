import os

import numpy as np
import torch

from common import Model, load_songs, BATCH_SIZE

cwd = os.path.dirname(__file__)


def assert_gpu_available():
    assert torch.cuda.is_available(), "GPU is not available"


def vectorize_string(string, character_to_index):
    return np.array([character_to_index[character] for character in string])


def get_batch(vectorized_songs, seq_length, batch_size):
    random_indexes = np.random.choice(vectorized_songs.shape[0] - seq_length - 1, batch_size)
    x_batch = np.reshape([vectorized_songs[i:i + seq_length] for i in random_indexes], [batch_size, seq_length])
    y_batch = np.reshape([vectorized_songs[i + 1:i + seq_length + 1] for i in random_indexes], [batch_size, seq_length])
    return x_batch, y_batch


def train():
    songs = load_songs()
    songs_joined = "\n\n".join(songs)
    vocabulary = sorted(set(songs_joined))
    char_to_index = {u: i for i, u in enumerate(vocabulary)}
    vectorized_songs = vectorize_string(songs_joined, char_to_index)

    model = Model(len(vocabulary), 256, 1024)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    for epoch in range(800):
        input_batch, target_batch = get_batch(vectorized_songs, seq_length=100, batch_size=BATCH_SIZE)

        total_loss = 0
        for i in range(len(input_batch)):
            model.zero_grad()
            prediction = model(torch.tensor(input_batch[i]))
            loss = loss_fn(prediction.cpu(), torch.from_numpy(target_batch[i]).long())

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(epoch, total_loss / len(input_batch))
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(cwd, "models", "model_" + str(epoch) + ".pt"))
            print("Model has been saved")

    print("Training finished")


train()
