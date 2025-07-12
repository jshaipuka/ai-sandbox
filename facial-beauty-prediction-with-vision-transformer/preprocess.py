import os
import sys

import cv2
import numpy as np

import common


def _eyes_centers(landmark):
    centers = []
    for start in [36, 42]:
        eye = landmark[0][start:start + 6]
        centers.append(np.take(eye, [1, 2, 4, 5], axis=0).mean(axis=0))
    return centers


# TODO: DRY
def face_align_dt_land(img, features, eyes_centers):
    # eyes center
    left_eye = eyes_centers[0]
    right_eye = eyes_centers[1]

    # Rotation Angle
    tg_a = (right_eye[1] - left_eye[1]) / (right_eye[0] - left_eye[0])
    ang = np.arctan(tg_a)
    angle = ang * (180 / np.pi)

    # Rotate the image
    num_rows, num_cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), angle, 1)
    img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))

    # Rotate the landmarks
    land_rot = np.empty([68, 2])
    for i in range(68):
        land_rot[i] = np.sum(rotation_matrix * np.transpose(np.append(features[i], 1)), axis=1)

    # Rotate the centers
    left_rot = np.sum(rotation_matrix * np.transpose(np.append(left_eye, 1)), axis=1)
    right_rot = np.sum(rotation_matrix * np.transpose(np.append(right_eye, 1)), axis=1)

    # Box Boundaries
    min_eye_y = np.min([left_rot[1], right_rot[1]])
    min_x = np.min(land_rot[0:, 0])
    max_x = np.max(land_rot[0:, 0])
    max_y = np.max(land_rot[0:, 1])

    Dist = max_y - min_eye_y
    min_y = min_eye_y - 0.6 * Dist
    if min_y < 0:
        min_y = 0

    img_cropped = img_rotation[int(np.round(min_y)):int(np.round(max_y)), int(np.round(min_x)):int(np.round(max_x))]

    img_res = cv2.resize(img_cropped, (224, 224), interpolation=cv2.INTER_AREA)
    return img_res


def preprocess(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    haarcascade = os.path.join(common.models_dir(), "haarcascade_frontalface_alt2.xml")
    detector = cv2.CascadeClassifier(haarcascade)
    faces = detector.detectMultiScale(image_gray)
    print("Faces:\n", faces)

    landmark_detector = cv2.face.createFacemarkLBF()
    lbf_model = os.path.join(common.models_dir(), "LBFmodel.yaml")
    landmark_detector.loadModel(lbf_model)

    _, landmarks = landmark_detector.fit(image_gray, faces)
    landmark = landmarks[0]
    eyes_centers = _eyes_centers(landmark)

    return face_align_dt_land(image, landmark[0], eyes_centers)


if __name__ == "__main__":
    path = sys.argv[1]
    image = cv2.imread(path)
    aligned_image = preprocess(image)
    cv2.imwrite("output.png", aligned_image)