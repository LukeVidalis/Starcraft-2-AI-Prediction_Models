from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Activation
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv3D

import numpy as np
import pylab as plt


img_width = 128
img_height = 128
rgb = 3

model = Sequential()

model.add(Conv2D(64, (3, 3), padding="same", input_shape=(128, 128, 3)))
model.add(Activation("relu"))

model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


seq = Sequential()
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), input_shape=(None, 40, 40, 1), padding="same",
                   return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding="same", return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding="same", return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding="same", return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same", data_format="channels_last"))
seq.compile(loss="binary_crossentropy", optimizer="adadelta")


