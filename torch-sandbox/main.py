import os

import keras_core as keras
import numpy as np
import regex as re
import torch
from torch.nn.functional import softmax, log_softmax
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


class SongsGenerator(torch.nn.Module):

    def __init__(self, batch_size, vocabulary_size, embedding_dim, hidden_dim):
        super(SongsGenerator, self).__init__()
        self.batch_size = batch_size
        self.embedding = torch.nn.Embedding(vocabulary_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = torch.nn.Linear(hidden_dim, vocabulary_size)

    def forward(self, sequences):
        embedded = self.embedding(sequences)
        prediction, _ = self.lstm(embedded)
        scores = self.linear(prediction)
        return log_softmax(scores, dim=2)  # (64, 100, 83)


def generate_text(model, char_to_index, index_to_char, start_string, generation_length=1000):
    input_eval = [char_to_index[s] for s in start_string]

    text_generated = []

    tqdm._instances.clear()

    for i in tqdm(range(generation_length)):
        predictions = torch.squeeze(model(torch.unsqueeze(torch.tensor(input_eval), 0)), 0)
        predicted_index = torch.multinomial(softmax(predictions, dim=0), 1, replacement=True)[-1].item()
        input_eval = [predicted_index]
        text_generated.append(index_to_char[predicted_index])

    return start_string + ''.join(text_generated)


def load_model(vocabulary_size, file_name):
    model = SongsGenerator(1, vocabulary_size, 256, 1024)
    model.load_state_dict(torch.load(os.path.join(cwd, "models", file_name)))
    model.eval()
    return model


def main():
    songs = load_songs()
    songs_joined = "\n\n".join(songs)
    vocabulary = sorted(set(songs_joined))
    char_to_index = {u: i for i, u in enumerate(vocabulary)}
    index_to_char = np.array(vocabulary)

    trained_model = load_model(len(vocabulary), "example_model_700_no_keras_with_batches.pt")
    print(generate_text(trained_model, char_to_index, index_to_char, "X", 1000))

    # vectorized_songs = vectorize_string(songs_joined, char_to_index)
    #
    # model = SongsGenerator(BATCH_SIZE, len(vocabulary), 256, 1024)
    #
    # loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    # for epoch in range(800):
    #     input_batch, target_batch = get_batch(vectorized_songs, seq_length=100, batch_size=BATCH_SIZE)
    #
    #     model.zero_grad()
    #     prediction = model(torch.tensor(input_batch))
    #     loss = loss_fn(prediction.permute((0, 2, 1)).cpu(), torch.from_numpy(target_batch).long())
    #
    #     loss.backward()
    #     optimizer.step()
    #
    #     print(epoch, loss.item())
    #     if epoch and epoch % 100 == 0:
    #         torch.save(model.state_dict(), os.path.join(cwd, "models", "model_" + str(epoch) + ".pt"))
    #         print("Model has been saved")
    #
    # print(generate_text(model, char_to_index, index_to_char, 'X'))


if __name__ == '__main__':
    main()
