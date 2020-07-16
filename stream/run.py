from imutils.video import VideoStream
import numpy as np
from stream.utils import *
from stream.face_detection import DetectFace
from stream.emotion_detection import EmotionDetect

params = get_face_model()
FaceDetector = DetectFace(params)
EmotionDetector = EmotionDetect(emotion_model)

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

while True:
    frame = vs.read()

    faces = FaceDetector.get_faces_from_frame(frame)

    for face in faces:
        (startX, startY, endX, endY, confidence) = face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[startY:endY, startX:endX]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        emotion, e_confidence = EmotionDetector.PredictEmotion(cropped_img)

        text = '{} {:.2f}'.format(emotion, e_confidence*100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), COLOR_BLUE, THICKNESS)
        cv2.putText(frame, text, (startX, y), FONT, .45, COLOR_BLUE, THICKNESS)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

vs.stop()
cv2.destroyAllWindows()
