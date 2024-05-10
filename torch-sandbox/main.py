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


def load_training_data():
    with open(os.path.join(cwd, "data", "irish.abc"), "r") as f:
        text = f.read()
    return extract_song_snippet(text)


def vectorize_string(string, character_to_index):
    return np.array([character_to_index[character] for character in string])


def get_batch(vectorized_songs, seq_length, batch_size):
    # the length of the vectorized songs string
    n = vectorized_songs.shape[0] - 1
    # randomly choose the starting indices for the examples in the training batch
    idx = np.random.choice(n - seq_length, batch_size)

    input_batch = [vectorized_songs[i:i + seq_length] for i in idx]
    output_batch = [vectorized_songs[i + 1:i + seq_length + 1] for i in idx]

    # x_batch, y_batch provide the true inputs and targets for network training
    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])
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


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    assert_gpu_available()
    print("Ready to go")
    torch.cuda.device_count()
    # m = torch.nn.LogSoftmax(dim=-1)
    # print(softmax(torch.Tensor([0, 0, 0, 1])))

    # model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # data = torch.rand(1, 3, 64, 64)
    # labels = torch.rand(1, 1000)
    # prediction = model(data)  # forward pass
    # loss = (prediction - labels).sum()
    # loss.backward()  # backward pass
    # print("Done")

    songs = load_training_data()
    example_song = songs[0]

    # Join our list of song strings into a single string containing all songs
    songs_joined = "\n\n".join(songs)

    # Find all unique characters in the joined string
    vocab = sorted(set(songs_joined))
    print("There are", len(vocab), "unique characters in the dataset")

    char2idx = {u: i for i, u in enumerate(vocab)}

    idx2char = np.array(vocab)

    # model = build_model(len(vocab), embedding_dim=256, rnn_units=1024, batch_size=BATCH_SIZE)
    # model.load_state_dict(torch.load(os.path.join(cwd, "models", "example_model_1400.pt"), map_location='cpu'))
    # model.eval()

    print('{')
    for char, _ in zip(char2idx, range(20)):
        print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
    print('  ...\n}')

    vectorized_songs = vectorize_string(songs_joined, char2idx)

    print(vectorized_songs)

    print('{} ---- characters mapped to int ----> {}'.format(repr(songs_joined[:10]), vectorized_songs[:10]))

    x_batch, y_batch = get_batch(vectorized_songs, seq_length=5, batch_size=BATCH_SIZE)

    for i, (input_idx, target_idx) in enumerate(zip(np.squeeze(x_batch), np.squeeze(y_batch))):
        print("Step {:3d}".format(i))
        print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
        print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

    model = build_model(len(vocab), embedding_dim=256, rnn_units=1024, batch_size=BATCH_SIZE)

    model.summary()

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')  # TODO: reduction
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    for i in range(1200):
        input_batch, target_batch = get_batch(vectorized_songs, seq_length=100, batch_size=BATCH_SIZE)

        prediction = model(input_batch)
        loss = loss_fn(prediction.permute((0, 2, 1)).cpu(), torch.from_numpy(target_batch).long())  # TODO: use CUDA

        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            print(i, loss.item())
        if i and i % 100 == 0:
            torch.save(model.state_dict(), os.path.join(cwd, "models", "model_" + str(i) + ".pt"))
            print("Model has been saved")

    print(generate_text(model, char2idx, idx2char, 'X'))
    x_answer, y_answer = get_batch(vectorized_songs, seq_length=100, batch_size=BATCH_SIZE)
    answer = model(x_answer)
    print("Input shape:      ", x_answer.shape, " # (batch_size, sequence_length)")
    print("Prediction shape: ", answer.shape, "# (batch_size, sequence_length, vocab_size)")
    # softmax(answer[0])
    # TODO: check if this is the correct replacement of tf.random.categorical
    # TODO: check softmax documentation, there is also log_softmax and torch.nn.LogSoftmax(dim=0)
    sampled_indices = torch.squeeze(torch.multinomial(softmax(answer[0], dim=0), 1, replacement=True)).cpu()
    print("Prediction: ", sampled_indices)

    print("Input: \n", repr("".join(idx2char[x_answer[0]])))
    print("Target: \n", repr("".join(idx2char[y_answer[0]])))
    print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))
