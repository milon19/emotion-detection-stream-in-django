import cv2
import os
from django.conf import settings

BASE_DIR = settings.BASE_DIR

prototxt = os.path.join(BASE_DIR, 'stream/face_detection_model/deploy.prototxt.txt')
face_model = os.path.join(BASE_DIR, 'stream/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel')
emotion_model = os.path.join(BASE_DIR, 'stream/emotion_detection_model/facial_expression_model_weights.h5')

COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
THICKNESS = 2

def get_face_model():
    params = {
        'prototxt': prototxt,
        'model': face_model
    }
    return params