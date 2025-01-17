import os

import cv2
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger()

def load_test_image():
    return cv2.cvtColor(cv2.imread(os.path.join(os.path.dirname(__file__), 'dataset', 'img1.jpg')), cv2.COLOR_BGR2RGB)

def test_gender():
    test_image = load_test_image()
    result = DeepFace.analyze(img_path=test_image, actions=['gender'], use_triton=True)

    print(result)


def test_verify():
    test_image = load_test_image()
    result = DeepFace.verify(img1_path=test_image, img2_path=test_image, use_triton=True)

    print(result)


def test_represent():
    test_image = load_test_image()
    result = DeepFace.represent(img_path=test_image, use_triton=True)

    print(result)
