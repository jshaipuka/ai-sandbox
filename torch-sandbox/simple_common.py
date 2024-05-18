import os

import keras_core as keras

cwd = os.path.dirname(__file__)

BATCH_SIZE = 50000


def build_model(batch_size):
    return keras.Sequential([
        keras.layers.Input(shape=(784,), batch_size=batch_size),
        keras.layers.Dense(units=16),
        keras.layers.Dense(units=16),
        keras.layers.Dense(units=10, activation="softmax")
    ])
