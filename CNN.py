from keras.models import Sequential
from keras.layers import Conv2D, Activation
from keras.optimizers import SGD, Adam

# Parameters
img_width = 128
img_height = 128
rgb = 3


def create_model(model_index):

    print("Creating Model")
    model = Sequential()

    if model_index == 1:
        model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=(img_width, img_height, rgb), padding="same"))

        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
        model.add(Activation("sigmoid"))

        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
        model.add(Activation("sigmoid"))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
        model.add(Activation("sigmoid"))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
        model.add(Activation("sigmoid"))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
        model.add(Activation("sigmoid"))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
        model.add(Activation("sigmoid"))

        model.add(Conv2D(filters=3, kernel_size=(3, 3), activation="sigmoid", padding="same"))

        opt = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
        model.compile(loss="mean_squared_error", optimizer=opt, metrics=['acc', 'mae'])

    elif model_index == 2:
        model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=(img_width, img_height, rgb), padding="same"))

        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
        model.add(Activation("relu"))

        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
        model.add(Activation("relu"))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
        model.add(Activation("relu"))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
        model.add(Activation("relu"))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
        model.add(Activation("relu"))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
        model.add(Activation("relu"))

        model.add(Conv2D(filters=3, kernel_size=(3, 3), activation="relu", padding="same"))

        opt = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
        model.compile(loss="mean_squared_error", optimizer=opt, metrics=['acc', 'mae'])

    return model




