import os

import numpy as np
import regex as re
import torch
from torch.nn.functional import softmax, log_softmax

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

    def __init__(self, vocabulary_size, embedding_dim, hidden_dim):
        super(SongsGenerator, self).__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)
        self.linear = torch.nn.Linear(hidden_dim, vocabulary_size)
        pass

    def forward(self, sequence, hidden):
        embedded = self.embedding(sequence)
        prediction, hidden = self.lstm(embedded.view(len(sequence), 1, -1), hidden)
        scores = self.linear(prediction.view(len(sequence), -1))
        return log_softmax(scores, dim=-1), hidden


def generate_text(model, char_to_index, index_to_char, start_string, generation_length=1000):
    c_0, h_0 = torch.zeros(1, 1, 1024), torch.zeros(1, 1, 1024)
    hidden = (h_0, c_0)
    input_eval = [char_to_index[s] for s in start_string]

    text_generated = []

    for i in range(generation_length):
        predictions, hidden = model(sequence=torch.tensor(input_eval), hidden=hidden)
        predicted_index = torch.multinomial(softmax(torch.squeeze(predictions, 0), dim=0), 1, replacement=True)[-1].item()
        input_eval = [predicted_index]
        text_generated.append(index_to_char[predicted_index])
        if i % 10 == 0:
            print("Predicted character", i)

    return start_string + ''.join(text_generated)


def load_model(vocabulary_size, file_name):
    model = SongsGenerator(vocabulary_size, 256, 1024)
    model.load_state_dict(torch.load(os.path.join(cwd, "models", file_name)))
    model.eval()
    return model


def main():
    # batch_size = 3
    # seq_length = 5
    # batch = torch.tensor([i for i in range(batch_size * seq_length)]).view((batch_size, seq_length))
    # print(batch)
    # for step in range(batch.shape[1] - 1):
    #     x, y = batch[:, step], batch[:, step + 1]
    #     print(x, y)
    songs = load_songs()
    songs_joined = "\n\n".join(songs)
    vocabulary = sorted(set(songs_joined))
    char_to_index = {u: i for i, u in enumerate(vocabulary)}
    index_to_char = np.array(vocabulary)

    trained_model = load_model(len(vocabulary), "main_model_700_no_keras.pt")
    print(generate_text(trained_model, char_to_index, index_to_char, "X", 2000))

    # vectorized_songs = vectorize_string(songs_joined, char_to_index)
    #
    # model = SongsGenerator(len(vocabulary), 256, 1024)
    #
    # loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    # for epoch in range(800):
    #     input_batch, target_batch = get_batch(vectorized_songs, seq_length=100, batch_size=BATCH_SIZE)
    #
    #     total_loss = 0
    #     for i in range(len(input_batch)):
    #         model.zero_grad()
    #         prediction = model(torch.tensor(input_batch[i]))
    #         loss = loss_fn(prediction.cpu(), torch.from_numpy(target_batch[i]).long())
    #
    #         loss.backward()
    #         optimizer.step()
    #
    #         total_loss += loss.item()
    #
    #     print(epoch, total_loss / len(input_batch))
    #     if (epoch + 1) % 100 == 0:
    #         torch.save(model.state_dict(), os.path.join(cwd, "models", "model_" + str(epoch) + ".pt"))
    #         print("Model has been saved")
    #
    # print(generate_text(model, char_to_index, index_to_char, 'X'))


if __name__ == '__main__':
    main()