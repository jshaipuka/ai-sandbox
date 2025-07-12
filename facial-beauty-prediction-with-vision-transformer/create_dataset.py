import argparse
import os
import shutil
import struct
import zipfile
from collections import defaultdict

import cv2
import numpy
import numpy as np
import openpyxl
import torch
from openpyxl.worksheet.worksheet import Worksheet
from tqdm import tqdm

from utils import face_align_dt_land

SCORERS_COUNT = 60


def _read_points(land_read):
    with open(land_read, 'rb') as f:
        data = f.read()
        points = struct.unpack('i172f', data)
        return np.reshape(np.array(points)[1:], (-1, 2))


def _create_dataset(path_to_images, path_to_facial_landmarks, image_to_mean_and_std_score):
    outputs_dir = _outputs_dir()
    x = []
    y = []
    sigma = []

    for image in tqdm(image_to_mean_and_std_score):
        try:
            mean, std = image_to_mean_and_std_score[image]
            full_path_image = os.path.join(path_to_images, image)
            img = cv2.imread(full_path_image)
            land_read = os.path.join(path_to_facial_landmarks, image[:-3] + 'pts')
            vec = _read_points(land_read)
            cropped_face = face_align_dt_land(img, vec, (224, 224))
            image = cropped_face.transpose((2, 0, 1))
            x.append(np.array(image))
            y.append(mean)
            sigma.append(std)
        except:
            print(f'Skipping {image} due to an exception')

    # TODO: in the original repository the sigma is always train sigma, check if it's correct
    training = (torch.tensor(numpy.array(x), dtype=torch.float32), torch.tensor(numpy.array(y), dtype=torch.float32), torch.tensor(numpy.array(sigma), dtype=torch.float32))

    dataset_dir = os.path.join(outputs_dir, "dataset")
    os.makedirs(dataset_dir)
    dataset_file = os.path.join(dataset_dir, f'dataset.pt')
    print(f'Saving dataset to {dataset_file}')
    torch.save(training, dataset_file)


def _image_to_score(data_scores: Worksheet):
    image_to_scores = defaultdict(lambda: numpy.zeros(SCORERS_COUNT))
    i = 0
    for row in data_scores.iter_rows(min_row=2, values_only=True):
        scorer = int(row[0])
        image_file = row[1]
        score = row[2]
        image_to_scores[image_file][scorer - 1] = float(score)
        i += 1
    print(f'Found {i} rows')
    return image_to_scores


def _outputs_dir():
    return os.path.join(os.getcwd(), "outputs")


def create_train_test_files():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path of the ZIP archive')
    opt = parser.parse_args()
    outputs_dir = _outputs_dir()

    if os.path.exists(outputs_dir) and os.path.isdir(outputs_dir):
        print('"outputs" directory already exists, recreating')
        shutil.rmtree(outputs_dir)
        os.makedirs(outputs_dir)

    print(f'Extracting {opt.data_path} to {outputs_dir}')
    with zipfile.ZipFile(opt.data_path, 'r') as zip_ref:
        zip_ref.extractall(outputs_dir)

    path_scores = os.path.join(outputs_dir, 'SCUT-FBP5500_v2/All_Ratings.xlsx')
    print(f'Reading facial beauty scores from {path_scores}')
    input_data_scores = openpyxl.load_workbook(path_scores)
    data_scores = input_data_scores.worksheets[0]
    print(f'Converting the facial beauty scores from {path_scores} to a dictionary')
    image_to_score = _image_to_score(data_scores)
    print(f'Conversion finished, dictionary length is {len(image_to_score)}')
    image_to_mean_and_std_score = {}
    for image in image_to_score:
        image_to_mean_and_std_score[image] = (image_to_score[image].mean(), image_to_score[image].std())
    print(f'Image to mean and std score calculated')

    print(f'Reading faces from {path_scores}')
    database_path = os.path.join(outputs_dir, 'SCUT-FBP5500_v2/Images')
    print(f'Reading facial landmarks from {path_scores}')
    land_path = os.path.join(outputs_dir, 'SCUT-FBP5500_v2/facial landmark')
    _create_dataset(database_path, land_path, image_to_mean_and_std_score)


if __name__ == "__main__":
    create_train_test_files()
