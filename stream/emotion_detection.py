from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Flatten,
                                     Conv2D, MaxPooling2D, AveragePooling2D)
import numpy as np


class EmotionDetect:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.get_model()

    def get_model(self):
        model = Sequential()

        # 1st convolution layer
        model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
        model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

        # 2nd convolution layer
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        # 3rd convolution layer
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Flatten())

        # fully connected neural networks
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(7, activation='softmax'))

        print('[INFO] loading emotion detection model...')
        model.load_weights(self.model_path)
        return model

    def PredictEmotion(self, Image):
        emotion_dict = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'sad', 5:'surprise', 6:'neutral'}
        prediction = self.model.predict(Image)
        max_index = int(np.argmax(prediction))
        return emotion_dict[max_index], np.max(prediction), prediction
