import cv2
import numpy as np


class DetectFace:
    def __init__(self, params):
        self.prototxt = params['prototxt']
        self.model = params['model']
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)

    def get_faces_from_frame(self, frame):
        (h, w) = frame.shape[0:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < .50:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            faces.append((startX, startY, endX, endY, confidence))

        return faces