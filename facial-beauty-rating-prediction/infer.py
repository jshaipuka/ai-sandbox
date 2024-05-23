import os
import sys

import keras_core as keras
import numpy as np
from PIL import Image

from common import cwd


def infer():
    # TODO: try using part of the data loaded from the archive to test the model
    image_path = sys.argv[1]
    print("Going to read image from", image_path)
    image = Image.open(image_path).convert("RGB")
    # TODO: resize
    pixels = np.expand_dims(np.array(image), axis=0) / 255.0

    model = keras.models.load_model(os.path.join(cwd, "model.keras"))
    model.summary()

    predictions = model.predict(pixels)

    print(predictions)


if __name__ == "__main__":
    infer()
