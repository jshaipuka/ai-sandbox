import os

import keras_core as keras
import matplotlib.pyplot as plt
import numpy as np

from common import cwd


def plot_image_prediction(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(np.squeeze(img), cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label, 100 * np.max(predictions_array), true_label), color=color)


def plot_value_prediction(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    this_plot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    this_plot[predicted_label].set_color('red')
    this_plot[true_label].set_color('blue')


def infer():
    _, (x_test, test_labels) = keras.datasets.mnist.load_data()
    test_images = np.expand_dims(x_test, axis=-1) / 255.0

    model = keras.models.load_model(os.path.join(cwd, "model_convolutional.keras"))
    model.summary()

    predictions = model.predict(test_images)

    num_rows = 5
    num_cols = 4
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image_prediction(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_prediction(i, predictions, test_labels)
    plt.show()


if __name__ == '__main__':
    infer()
