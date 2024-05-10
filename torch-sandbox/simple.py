import math
import os
import tempfile

import keras_core as keras
import numpy as np
import torch
from PIL import Image
from torch.nn.functional import softmax

import mnist_loader

BATCH_SIZE = 50000

temp_dir_name = tempfile.mkdtemp()


def split_into_batches(array, batch_size):
    return np.array_split(array, range(batch_size, len(array), batch_size))


def build_model(batch_size):
    return keras.Sequential([
        keras.layers.Input(shape=(784,), batch_size=batch_size),
        keras.layers.Dense(units=16),
        keras.layers.Dense(units=16),
        keras.layers.Dense(units=10)
    ])


def as_image(image):
    return Image.fromarray((255 * np.reshape(image, (math.isqrt(image.size), -1))).astype('uint8'))


def show_digit(image):
    as_image(image).show()


def save_digit(file_name, image):
    as_image(image).save(os.path.join(temp_dir_name, file_name + ".png"))


def pick_max_p(distribution):
    return [np.argmax(distribution[index].cpu().detach()) for index in range(len(distribution))]


def sample(distribution):
    answer = torch.squeeze(torch.multinomial(distribution, 1, replacement=True))
    return answer.cpu().numpy()


print("Ready to go")

model = build_model(BATCH_SIZE)

model.summary()

data = mnist_loader.load_data()

training_data = data[0]
training_inputs = training_data[0]
training_targets = training_data[1]

loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

for i in range(5000):
    input_batch = np.reshape(training_inputs, [BATCH_SIZE, -1])
    target_batch = training_targets

    prediction = model(input_batch)
    loss = loss_fn(prediction, torch.LongTensor(target_batch).cuda())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if i % 10 == 0:
        print(i, loss.item())

validation_data = data[1]
validation_inputs = validation_data[0]
validation_targets = validation_data[1]
validation_input_batches = split_into_batches(validation_inputs, BATCH_SIZE)
validation_target_batches = split_into_batches(validation_targets, BATCH_SIZE)

wrong_recognition = []

for i in range(min(100, len(validation_input_batches))):
    input_batch = validation_input_batches[i]
    target_batch = validation_target_batches[i]
    prediction = model(input_batch)
    distribution = softmax(prediction, dim=1)
    answer_np = sample(distribution)
    right_answers = np.sum(answer_np == target_batch)
    for j in range(len(target_batch)):
        if target_batch[j] != answer_np[j]:
            wrong_recognition.append((j, target_batch[j], answer_np[j]))
    print("Got ", right_answers, " right answers out of ", len(target_batch))

print("Saving wrongly recognized digits to ", temp_dir_name)
for (index, expected, actual) in wrong_recognition:
    save_digit(str(index) + "_" + str(expected) + "_" + str(actual), validation_inputs[index])
