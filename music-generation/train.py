import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common import GRUModel, load_songs, BATCH_SIZE, HIDDEN_DIM, SEQ_LENGTH, ModelType, DEFAULT_MODEL_TYPE, LSTMModel, determine_device

cwd = os.path.dirname(__file__)


def _vectorize_string(string, character_to_index):
    return np.array([character_to_index[character] for character in string])


def _get_batch(vectorized_songs, seq_length, batch_size):
    indices = np.random.choice(vectorized_songs.shape[0] - seq_length - 1, batch_size)
    return np.stack([vectorized_songs[i:i + seq_length] for i in indices]), np.stack([vectorized_songs[i + 1:i + seq_length + 1] for i in indices])


def train():
    device = determine_device()
    print("Device is", device)
    print("Model is", DEFAULT_MODEL_TYPE)
    songs = load_songs()
    songs_joined = "\n\n".join(songs)
    vocabulary = sorted(set(songs_joined))
    char_to_index = {u: i for i, u in enumerate(vocabulary)}
    vectorized_songs = _vectorize_string(songs_joined, char_to_index)

    model = LSTMModel(len(vocabulary), 256, HIDDEN_DIM).to(device) \
        if DEFAULT_MODEL_TYPE == ModelType.LSTM \
        else GRUModel(len(vocabulary), 256, HIDDEN_DIM).to(device)

    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    for epoch in range(1600):
        input_batch, target_batch = _get_batch(vectorized_songs, seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE)

        h_0, c_0 = torch.zeros(1, BATCH_SIZE, HIDDEN_DIM).to(device), torch.zeros(1, BATCH_SIZE, HIDDEN_DIM).to(device)
        prediction, _ = model(torch.tensor(input_batch).to(device), (h_0, c_0) if DEFAULT_MODEL_TYPE == ModelType.LSTM else h_0)
        loss = loss_fn(prediction.permute(0, 2, 1), torch.from_numpy(target_batch).to(device).long())

        loss.backward()
        optimizer.step()
        model.zero_grad()

        if epoch % 10 == 0:
            print(epoch, loss.item())
        if (epoch + 1) % 100 == 0:
            file_name = "model_" + str(epoch + 1) + ".pt"
            torch.save(model.state_dict(), os.path.join(cwd, "models", file_name))
            print("Model has been saved as", file_name)

    print("Training finished")


if __name__ == "__main__":
    train()
