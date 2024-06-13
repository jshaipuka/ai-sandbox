import sys

import h5py
import keras_core as keras
import numpy as np
from matplotlib import pyplot as plt


class TrainingDatasetLoader(object):
    def __init__(self, data_path):

        print("Opening {}".format(data_path))
        sys.stdout.flush()

        self.cache = h5py.File(data_path, 'r')

        print("Loading data into memory...")
        sys.stdout.flush()
        self.images = self.cache['images'][:]
        self.labels = self.cache['labels'][:].astype(np.float32)
        self.image_dims = self.images.shape
        n_train_samples = self.image_dims[0]

        self.train_idx = np.random.permutation(np.arange(n_train_samples))

        self.pos_train_idx = self.train_idx[self.labels[self.train_idx, 0] == 1.0]
        self.neg_train_idx = self.train_idx[self.labels[self.train_idx, 0] != 1.0]

    def get_train_size(self):
        return self.train_idx.shape[0]

    def get_train_steps_per_epoch(self, batch_size, factor=10):
        return self.get_train_size() // factor // batch_size

    def get_batch(self, n, only_faces=False, p_pos=None, p_neg=None, return_idx=False):
        if only_faces:
            selected_idx = np.random.choice(self.pos_train_idx, size=n, replace=False, p=p_pos)
        else:
            selected_pos_idx = np.random.choice(self.pos_train_idx, size=n // 2, replace=False, p=p_pos)
            selected_neg_idx = np.random.choice(self.neg_train_idx, size=n // 2, replace=False, p=p_neg)
            selected_idx = np.concatenate((selected_pos_idx, selected_neg_idx))

        sorted_idx = np.sort(selected_idx)
        train_img = (self.images[sorted_idx, :, :, ::-1] / 255.).astype(np.float32)
        train_label = self.labels[sorted_idx, ...]
        return (train_img, train_label, sorted_idx) if return_idx else (train_img, train_label)

    def get_n_most_prob_faces(self, prob, n):
        idx = np.argsort(prob)[::-1]
        most_prob_idx = self.pos_train_idx[idx[:10 * n:10]]
        return (self.images[most_prob_idx, ...] / 255.).astype(np.float32)

    def get_all_train_faces(self):
        return self.images[self.pos_train_idx]


BATCH_SIZE = 64
EPOCHS = 5


def train():
    path_to_training_data = keras.utils.get_file('train_face.h5', 'https://www.dropbox.com/s/hlz8atheyozp1yx/train_face.h5?dl=1')
    loader = TrainingDatasetLoader(path_to_training_data)
    number_of_training_examples = loader.get_train_size()
    print(number_of_training_examples)
    (images, labels) = loader.get_batch(1000)
    face_images = images[np.where(labels == 1)[0]]
    not_face_images = images[np.where(labels == 0)[0]]
    idx_face = 24
    idx_not_face = 180

    plt.figure(figsize=(5, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(face_images[idx_face])
    plt.title("Face")
    plt.grid(False)

    plt.subplot(1, 2, 2)
    plt.imshow(not_face_images[idx_not_face])
    plt.title("Not Face")
    plt.grid(False)

    plt.show()


if __name__ == "__main__":
    train()
