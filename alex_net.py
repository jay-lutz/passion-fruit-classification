from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout


class AlexNet:
    def __init__(self):
        self.model = Sequential([Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                                        input_shape=(227, 227, 3)),
                                 BatchNormalization(),
                                 MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
                                 Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu',
                                        padding='same'),
                                 BatchNormalization(),
                                 MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
                                 Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                        padding='same'),
                                 BatchNormalization(),
                                 Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                        padding='same'),
                                 BatchNormalization(),
                                 Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                        padding='same'),
                                 BatchNormalization(),
                                 MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
                                 Flatten(),
                                 Dense(4096, activation='relu'),
                                 Dropout(0.5),
                                 Dense(4096, activation='relu'),
                                 Dropout(0.5),
                                 Dense(3, activation='softmax')
                                 ])

    def alex_net_model(self):
        return self.model
