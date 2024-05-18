import math
import os
import tempfile

import keras_core as keras
import numpy as np
import torch
from PIL import Image

import mnist_loader
from simple_common import BATCH_SIZE

cwd = os.path.dirname(__file__)

temp_dir_name = tempfile.mkdtemp()


def split_into_batches(array, batch_size):
    return np.array_split(array, range(batch_size, len(array), batch_size))


def as_image(image):
    return Image.fromarray((255 * np.reshape(image, (math.isqrt(image.size), -1))).astype('uint8'))


def save_digit(file_name, image):
    as_image(image).save(os.path.join(temp_dir_name, file_name + ".png"))


def pick_max_probability(distribution):
    return [np.argmax(distribution[index].cpu().detach()) for index in range(len(distribution))]


def sample_probability_distribution(distribution):
    answer = torch.squeeze(torch.multinomial(distribution, 1, replacement=True))
    return answer.cpu().numpy()


print("Ready to go")

model = keras.models.load_model(os.path.join(cwd, "models", "simple_model.keras"))

model.summary()

data = mnist_loader.load_data()

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
    answer_np = sample_probability_distribution(prediction)
    right_answers = np.sum(answer_np == target_batch)
    for j in range(len(target_batch)):
        if target_batch[j] != answer_np[j]:
            wrong_recognition.append((j, target_batch[j], answer_np[j]))
    print("Got ", right_answers, " right answers out of ", len(target_batch))

print("Saving wrongly recognized digits to ", temp_dir_name)
for (index, expected, actual) in wrong_recognition:
    save_digit(str(index) + "_" + str(expected) + "_" + str(actual), validation_inputs[index])
