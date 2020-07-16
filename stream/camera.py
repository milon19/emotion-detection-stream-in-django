from imutils.video import VideoStream
import numpy as np
from stream.emotion_detection import EmotionDetect
from stream.utils import *
from stream.face_detection import DetectFace

params = get_face_model()
FaceDetector = DetectFace(params)

EmotionDetector = EmotionDetect(emotion_model)

class openCamera(object):
    def __init__(self):
        self.cap = VideoStream(src=0).start()
        self.params = {}

    def __del__(self):
        self.cap.stop()
        self.cap.stream.release()

    def get_frame_web(self):
        frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        faces = FaceDetector.get_faces_from_frame(frame)

        for face in faces:
            (startX, startY, endX, endY, confidence) = face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_gray = gray[startY:endY, startX:endX]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            emotion, e_confidence, prediction = EmotionDetector.PredictEmotion(cropped_img)

            self.params = {
                'emotion': emotion,
                'e_confidence': e_confidence,
                'prediction': prediction
            }

            text = '{} {:.2f}'.format(emotion, e_confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLOR_BLUE, THICKNESS)
            cv2.putText(frame, text, (startX, y), FONT, .45, COLOR_BLUE, THICKNESS)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def get_frame_raw(self):
        frame = self.cap.read()
        return frame

    def gen(cam):
        while True:
            frame = cam.get_frame_web()
            yield b'--frame\r\n 'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'

    def get_params(self):
        return self.params
