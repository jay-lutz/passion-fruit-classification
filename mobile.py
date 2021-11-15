from tensorflow.keras.applications import MobileNetV3Large
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout


class MobileNetModified:
    def __init__(self):
        self.model = Sequential()
        base = MobileNetV3Large(include_top=False, weights=None, include_preprocessing=True)
        self.model.add(base)
        self.model.add(Flatten())
        self.model.add(Dense(1028, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1028, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(3, activation='softmax'))

    def get_model(self):
        return self.model
