import os
import sys
from os import listdir
from os.path import isfile

import keras_core as keras
import numpy as np
from PIL import Image

from common import cwd


def infer():
    # TODO: try using part of the data loaded from the archive to test the model
    image_folder = sys.argv[1]
    print("Going to read images from", image_folder)
    paths = [os.path.join(image_folder, f) for f in listdir(image_folder)]
    files = [p for p in paths if isfile(os.path.join(image_folder, p))]
    images = np.array([np.array(Image.open(f).convert("RGB")) / 255.0 for f in files])
    model = keras.models.load_model(os.path.join(cwd, "model.keras"))
    model.summary()
    predictions = model.predict(np.array(images))
    print(list(zip(paths, np.squeeze(predictions))))


if __name__ == "__main__":
    infer()
