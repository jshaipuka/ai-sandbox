import os
import sys

import keras_core as keras
import numpy as np

from common import cwd

BATCH_SIZE = 64
EPOCHS = 6


def train():
    archive_path = sys.argv[1]
    print("Going to read data from", archive_path)
    data = np.load(archive_path)
    images, scores = data["X"] / 255.0, np.asarray(data["y"], dtype=float)
    training_length = 4900
    training_indices = np.random.choice(len(images), training_length, replace=False)
    validation_indices = np.delete(np.arange(len(images)), training_indices)
    training_images, training_scores = np.take(images, training_indices, axis=0), np.take(scores, training_indices, axis=0)
    validation_images, validation_scores = np.take(images, validation_indices, axis=0), np.take(scores, validation_indices, axis=0)

    model = keras.Sequential([
        keras.Input(shape=(350, 350, 3)),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
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
