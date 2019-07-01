import keras
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras import backend as K
import os

img_width = 128
img_height = 128
smooth = 1

def create_conv_layer(features, stride, activationfn, padding, prevLayer, dropout):

    conv_layer = Conv2D(features, stride, activation=activationfn, padding=padding)(prevLayer)
    conv_layer = Dropout(dropout)(conv_layer)
    conv_layer = Conv2D(features, stride, activation=activationfn, padding=padding)(conv_layer)

    return conv_layer

