import os
import sys

import keras_core as keras
import numpy as np
from PIL import Image

from common import cwd

BATCH_SIZE = 64
EPOCHS = 6


def _training_validation_test_indices(training_length, validation_length, test_length):
    shuffled_indices = np.arange(training_length + validation_length + test_length)
    np.random.shuffle(shuffled_indices)
    [training_indices, validation_indices, test_indices] = np.split(shuffled_indices, [training_length, training_length + validation_length])
    return training_indices, validation_indices, test_indices


def train():
    archive_path = sys.argv[1]
    print("Going to read data from", archive_path)
    data = np.load(archive_path)
    images, scores = data["X"] / 255.0, np.asarray(data["y"], dtype=float)
    training_indices, validation_indices, test_indices = _training_validation_test_indices(4900, 500, 100)
    assert len(images) == len(scores) == len(training_indices) + len(validation_indices) + len(test_indices), "Samples, labels or split length mismatch"
    training_images, training_scores = np.take(images, training_indices, axis=0), np.take(scores, training_indices, axis=0)
    validation_images, validation_scores = np.take(images, validation_indices, axis=0), np.take(scores, validation_indices, axis=0)
    test_images, test_scores = np.take(images, test_indices, axis=0), np.take(scores, test_indices, axis=0)

    test_data_folder = os.path.join(os.path.dirname(archive_path), "test")
    print("Going to save test data to", test_data_folder)
    for i, image_bytes in enumerate(test_images):
        score = test_scores[i]
        image = Image.fromarray(np.asarray(image_bytes * 255).astype(np.uint8), "RGB")
        image.save(os.path.join(test_data_folder, "{:04d}".format(i) + "_" + str(score).replace(".", "_") + ".png"))

    model = keras.Sequential([
        keras.Input(shape=(350, 350, 3)),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu"),
        keras.layers.MaxPool2D(pool_size=(3, 3)),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation="relu"),
        keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation="relu"),
        keras.layers.MaxPool2D(pool_size=(3, 3)),
        keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation="relu"),
        keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation="relu"),
        keras.layers.Conv2D(filters=64, kernel_size=(7, 7), activation="relu"),
        keras.layers.MaxPool2D(pool_size=(3, 3)),
        keras.layers.Conv2D(filters=64, kernel_size=(7, 7), activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1, activation="linear")
    ])
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae", "mape", "cosine_similarity"])
    model.fit(training_images, training_scores, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(validation_images, validation_scores))

    file_name = os.path.join(cwd, "model.keras")
    model.save(os.path.join(cwd, file_name))
    print("The model has been saved as", file_name)


if __name__ == "__main__":
    train()
