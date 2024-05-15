import os

import numpy as np
import torch

from common import Model, load_songs, BATCH_SIZE, HIDDEN_DIM

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

    model = Model(len(vocabulary), 256, HIDDEN_DIM)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    for epoch in range(1600):
        input_batch, target_batch = get_batch(vectorized_songs, seq_length=100, batch_size=BATCH_SIZE)  # (64, 100)

        model.zero_grad()
        h_0, c_0 = torch.zeros(1, BATCH_SIZE, HIDDEN_DIM), torch.zeros(1, BATCH_SIZE, HIDDEN_DIM)
        hidden = (h_0, c_0)
        prediction, _ = model(torch.tensor(input_batch), hidden)
        loss = loss_fn(prediction.view(BATCH_SIZE * 100, -1).cpu(), torch.squeeze(torch.from_numpy(target_batch).view(BATCH_SIZE * 100, -1)).long())  # (64, 100, 83), (64, 100)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(epoch, loss.item())
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(cwd, "models", "model_" + str(epoch) + ".pt"))
            print("Model has been saved")

    print("Training finished")


train()
