import keras_core as keras
import numpy as np

BATCH_SIZE = 64
EPOCHS = 5


def train():
    (x_train, train_labels), (x_test, test_labels) = keras.datasets.mnist.load_data()
    train_images = np.expand_dims(x_train, axis=-1) / 255.0
    test_images = np.expand_dims(x_test, axis=-1) / 255.0

    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.summary()
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-1), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Accuracy:", test_acc)


if __name__ == "__main__":
    train()
