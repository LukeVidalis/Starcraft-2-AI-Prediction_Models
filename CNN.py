from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Activation
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv3D
from process_array import *
import os
from settings import *

# Paths
json_file = os.path.join(WEIGHTS_DIR, 'CNN_model.json')

# Parameters
img_width = 128
img_height = 128
rgb = 3
epochs_num = 100
batch_size = 250


def load_files():
    data = load_array("Acid_Plant")
    return data['x'], data['Y']


def create_model():
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), input_shape=(img_width, img_height, rgb), padding="same",
                         return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", return_sequences=True))
    model.add(BatchNormalization())

    model.add(Conv3D(filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same",
                     data_format="channels_last"))
    model.compile(loss="binary_crossentropy", optimizer="adadelta")

    return model


def save_model(model):
    # TODO add weights file
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)


def load_model():
    """ code here """


def train_model(model, x, Y):

    model.fit(x=x, y=Y, batch_size=batch_size, epochs=epochs_num, verbose=1, callbacks=None, validation_split=0.2,
              validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
              steps_per_epoch=None, validation_steps=None, validation_freq=1)

    return model


if __name__ == "__main__":
    x, Y = load_files()
    model = create_model()
    model = train_model(model, x, Y)
