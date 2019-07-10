from keras.models import Sequential, load_model

from keras.layers import Conv2D, Dropout, MaxPooling2D, Activation
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv3D
from process_array import *
from keras.models import load_model
import os
from settings import *

# Paths
json_file = os.path.join(WEIGHTS_DIR, 'CNN_model.json')

# Parameters
img_width = 128
img_height = 128
rgb = 3
epochs_num = 100
batch_size = 1


def load_files():
    data = load_array("Acid_Plant.npz")
    return data['x'], data['Y']


def create_model():
    seq_model = Sequential()
    seq_model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(img_width, img_height, rgb), padding="same"))

    seq_model.add(BatchNormalization())

    seq_model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
    seq_model.add(BatchNormalization())

    seq_model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
    seq_model.add(BatchNormalization())

    seq_model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
    seq_model.add(BatchNormalization())

    seq_model.add(Conv2D(filters=3, kernel_size=(3, 3), activation="sigmoid", padding="same"))

    seq_model.compile(loss="binary_crossentropy", optimizer="adadelta")

    return seq_model


def save_model(seq_model):
    # TODO add weights file
    json_string = seq_model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)


def get_model():
    return load_model(json_file)


def train_model(seq_model, x, Y):

    history = seq_model.fit(x=x, y=Y, batch_size=batch_size, epochs=epochs_num, verbose=2, callbacks=None,
                            validation_split=0.2, validation_data=None, shuffle=True, class_weight=None,
                            sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)

    return history, seq_model

    # x: Input
    # y: Output
    # batch_size: number of samples per gradient update
    # epochs: number of epochs to train the model
    # verbose: Verbosity mode (0, 1 OR 2)
    # callbacks: callbacks to apply during training TODO: Look into callbacks
    # validation_split: Fraction of the training data to be used as validation data
    # validation_data: Data to perform validation on
    # shuffle:  shuffle the training data before each epoch
    # class_weight: dictionary for adding weight to different classes
    # sample_weight: array of sample weights
    # initial_epoch: epoch at which to start training. Useful for resuming a previous training run.
    # steps_per_epoch: total number of steps (batches of samples) before declaring one epoch finished and
    # starting the next epoch.
    # validation_steps: if steps_per_epoch != None, total number of steps to validate before stopping
    # validation_freq: run validation every x epochs. ( if validation_data != None)



if __name__ == "__main__":
    print("Getting Data")
    x, Y = load_files()
    print("\nCreating Model")
    seq_model = create_model()
    print("\nTraining Model")
    history, seq_model = train_model(seq_model, x, Y)
    print("\nSaving Model")
    save_model(seq_model)
