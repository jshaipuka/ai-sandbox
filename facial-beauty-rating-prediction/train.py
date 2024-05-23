import os
import sys

import keras_core as keras
import numpy as np

from common import cwd

BATCH_SIZE = 64
EPOCHS = 50


def train():
    archive_path = sys.argv[1]
    print("Going to read data from", archive_path)
    b = np.load(archive_path)
    x, labels = b["X"] / 255.0, np.asarray(b["y"], dtype=float)
    train_length = 4900
    train_images, train_labels = x[:train_length], labels[:train_length]
    test_images, test_labels = x[train_length:], labels[train_length:]

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
    model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)

    loss_and_metrics = model.evaluate(test_images, test_labels, return_dict=True)
    print("Loss and metrics of the model:", loss_and_metrics)

    file_name = "model.keras"
    model.save(os.path.join(cwd, file_name))


if __name__ == "__main__":
    train()
