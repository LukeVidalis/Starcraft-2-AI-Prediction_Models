from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dropout
from keras.layers.normalization import BatchNormalization

# Parameters
img_width = 128
img_height = 128
rgb = 3


def create_model():
    print("Creating Model")
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(7, 7), input_shape=(img_width, img_height, rgb), padding="same"))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
    model.add(Activation("sigmoid"))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
    model.add(Activation("sigmoid"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
    model.add(Activation("sigmoid"))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
    model.add(Activation("sigmoid"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
    model.add(Activation("sigmoid"))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
    model.add(Activation("sigmoid"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=3, kernel_size=(3, 3), activation="sigmoid", padding="same"))

    model.compile(loss="binary_crossentropy", optimizer="adadelta")

    return model




