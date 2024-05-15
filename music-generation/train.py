import os

import numpy as np
import torch

from common import Model, load_songs, BATCH_SIZE, HIDDEN_DIM, SEQ_LENGTH

cwd = os.path.dirname(__file__)


def vectorize_string(string, character_to_index):
    return np.array([character_to_index[character] for character in string])


def get_batch(vectorized_songs, seq_length, batch_size):
    random_indexes = np.random.choice(vectorized_songs.shape[0] - seq_length - 1, batch_size)
    x_batch = np.reshape([vectorized_songs[i:i + seq_length] for i in random_indexes], [batch_size, seq_length])
    y_batch = np.reshape([vectorized_songs[i + 1:i + seq_length + 1] for i in random_indexes], [batch_size, seq_length])
    return x_batch, y_batch


def train():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print("Device is", device)
    songs = load_songs()
    songs_joined = "\n\n".join(songs)
    vocabulary = sorted(set(songs_joined))
    char_to_index = {u: i for i, u in enumerate(vocabulary)}
    vectorized_songs = vectorize_string(songs_joined, char_to_index)

    model = Model(len(vocabulary), 256, HIDDEN_DIM).to(device)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    for epoch in range(1600):
        input_batch, target_batch = get_batch(vectorized_songs, seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE)

        model.zero_grad()
        h_0, c_0 = torch.zeros(1, BATCH_SIZE, HIDDEN_DIM).to(device), torch.zeros(1, BATCH_SIZE, HIDDEN_DIM).to(device)
        prediction, _ = model(torch.tensor(input_batch).to(device), (h_0, c_0))
        loss = loss_fn(prediction.view(BATCH_SIZE * SEQ_LENGTH, -1), torch.from_numpy(target_batch).to(device).view(BATCH_SIZE * SEQ_LENGTH).long())

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(epoch, loss.item())
        if (epoch + 1) % 100 == 0:
            file_name = "model_" + str(epoch) + ".pt"
            torch.save(model.state_dict(), os.path.join(cwd, "models", file_name))
            print("Model has been saved as", file_name)

    print("Training finished")


train()
