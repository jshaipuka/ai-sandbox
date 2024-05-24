import os
import sys
from os import listdir
from os.path import isfile

import keras_core as keras
import numpy as np
from PIL import Image

from common import cwd


def _extract_expected_score(f):
    name = f.split(".")[0]
    tokens = name.split("_")
    return float(tokens[1] + "." + tokens[2])


def infer():
    image_folder = sys.argv[1]
    print("Going to read images from", image_folder)
    files = [f for f in listdir(image_folder) if isfile(os.path.join(image_folder, f))]
    expected_scores = [_extract_expected_score(f) for f in files]
    full_paths = [os.path.join(image_folder, p) for p in files]
    images = np.array([np.array(Image.open(f).convert("RGB")) / 255.0 for f in full_paths])
    model = keras.models.load_model(os.path.join(cwd, "model.keras"))
    model.summary()
    predicted_scores = model.predict(np.array(images))
    for i, predicted_score in enumerate(np.squeeze(predicted_scores)):
        print("For image", files[i], "predicted score is", predicted_score, ", expected score is", expected_scores[i])


if __name__ == "__main__":
    infer()
