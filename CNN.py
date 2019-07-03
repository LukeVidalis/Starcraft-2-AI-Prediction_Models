import keras
from keras.models import Model, Sequential
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Dropout, MaxPooling2D, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras import backend as K
import os

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





