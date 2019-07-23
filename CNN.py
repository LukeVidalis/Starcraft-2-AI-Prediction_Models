from keras.models import Sequential, Model
from keras.layers import Conv2D, Activation, Dense, Conv2DTranspose, Input, BatchNormalization, Dropout, MaxPooling2D
from keras.optimizers import SGD, Adam

# Parameters
img_width = 128
img_height = 128
rgb = 3


def create_model(model_index):

    print("Creating Model")

    if model_index == 1:
        model = Sequential()

        model.add(Conv2D(filters=16, kernel_size=(3, 3, 3), input_shape=(img_width, img_height, rgb), padding="same"))

        model.add(Conv2D(filters=32, kernel_size=(3, 3, 3), padding="same"))
        model.add(Activation("sigmoid"))

        model.add(Conv2D(filters=32, kernel_size=(3, 3, 3), padding="same"))
        model.add(Activation("sigmoid"))

        model.add(Conv2D(filters=64, kernel_size=(3, 3, 3), padding="same"))
        model.add(Activation("sigmoid"))

        model.add(Conv2D(filters=64, kernel_size=(3, 3, 3), padding="same"))
        model.add(Activation("sigmoid"))

        model.add(Conv2D(filters=128, kernel_size=(3, 3, 3), padding="same"))
        model.add(Activation("sigmoid"))

        model.add(Conv2D(filters=128, kernel_size=(3, 3, 3), padding="same"))
        model.add(Activation("sigmoid"))

        model.add(Conv2D(filters=3, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"))

        # opt = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
        opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
        model.compile(loss="mean_absolute_error", optimizer=opt, metrics=['acc', 'mae'])

    elif model_index == 2:
        model = Sequential()

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

        opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
        model.compile(loss="mean_absolute_percentage_error", optimizer=opt, metrics=['acc', 'mae'])

    elif model_index == 3:
        #Input Layer with shape (128, 128, 3)
        input_layer = Input(shape=(128, 128, 3))
        in_norm = BatchNormalization()(input_layer)

        # Convolutional layers (Encoder)
        conv_1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), activation="relu", padding="same")(in_norm)
        conv_1_norm = BatchNormalization()(conv_1)
        conv_1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), activation="relu", padding="same")(conv_1_norm)
        conv_1_norm = BatchNormalization()(conv_1)
        pool_1 = MaxPooling2D()(conv_1_norm)

        conv_2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation="relu", padding="same")(pool_1)
        conv_2_norm = BatchNormalization()(conv_2)
        conv_2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation="relu", padding="same")(conv_2_norm)
        conv_2_norm = BatchNormalization()(conv_2)
        pool_2 = MaxPooling2D()(conv_2_norm)

        conv_3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same")(pool_2)
        conv_3_norm = BatchNormalization()(conv_3)
        conv_3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same")(conv_3_norm)
        conv_3_norm = BatchNormalization()(conv_3)
        pool_3 = MaxPooling2D()(conv_3_norm)
        drop_3 = Dropout(0.25)(pool_3)

        conv_4 = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same")(drop_3)
        conv_4_norm = BatchNormalization()(conv_4)
        conv_4 = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same")(conv_4_norm)
        conv_4_norm = BatchNormalization()(conv_4)
        pool_4 = MaxPooling2D()(conv_4_norm)
        drop_4 = Dropout(0.25)(pool_4)

        # Deconvolutional Layers (Decoder)
        deconv_1 = Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=(2, 2), output_padding= None, activation="relu", padding="same")(drop_4)
        deconv_1_norm = BatchNormalization()(deconv_1)

        # flatten = Flatten()(drop_4)
        hidden_layer_1 = Dense(1024, activation="relu")(deconv_1_norm)
        hl_1_norm = BatchNormalization()(hidden_layer_1)
        drop_1 = Dropout(0.3)(hl_1_norm)

        deconv_2 = Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=(2, 2), output_padding= None, activation="relu", padding="same")(drop_1)
        deconv_2_norm = BatchNormalization()(deconv_2)

        hidden_layer_2 = Dense(512, activation="relu")(deconv_2_norm)
        hl_2_norm = BatchNormalization()(hidden_layer_2)
        drop_2 = Dropout(0.3)(hl_2_norm)

        deconv_3 = Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=(2, 2), output_padding= None, activation="relu", padding="same")(drop_2)
        deconv_3_norm = BatchNormalization()(deconv_3)

        hidden_layer_3 = Dense(256, activation="relu")(deconv_3_norm)
        hl_3_norm = BatchNormalization()(hidden_layer_3)
        drop_3 = Dropout(0.3)(hl_3_norm)

        deconv_4 = Conv2DTranspose(filters=64, kernel_size=(7, 7), strides=(2, 2), output_padding= None, activation="relu", padding="same")(drop_3)
        deconv_4_norm = BatchNormalization()(deconv_4)

        output_layer = Dense(3, activation="relu")(deconv_4_norm)

        model = Model(outputs=output_layer, inputs=input_layer)

        opt = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
        model.compile(loss="mean_squared_error", optimizer=opt, metrics=['acc'])
        model.summary()

    elif model_index == 4:
        input_layer = Input(shape=(128, 128, 3))
        in_norm = BatchNormalization()(input_layer)

        conv_1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), activation="relu", padding="same")(in_norm)
        conv_1_norm = BatchNormalization()(conv_1)

        # flatten = Flatten()(conv_1_norm)
        hidden_layer_1 = Dense(1024, activation="relu")(conv_1_norm)
        hl_1_norm = BatchNormalization()(hidden_layer_1)
        drop_1 = Dropout(0.3)(hl_1_norm)

        output_layer = Dense(3, activation="relu")(drop_1)

        model = Model(outputs=output_layer, inputs=input_layer)
        model.summary()

    elif model_index == 5:
        #Input Layer with shape (128, 128, 3)
        input_layer = Input(shape=(128, 128, 3))
        in_norm = BatchNormalization()(input_layer)

        # Convolutional layers (Encoder)
        conv_1 = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), activation="relu", padding="same")(in_norm)
        conv_1_norm = BatchNormalization()(conv_1)
        conv_1 = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), activation="relu", padding="same")(conv_1_norm)
        conv_1_norm = BatchNormalization()(conv_1)

        conv_2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same")(conv_1_norm)
        conv_2_norm = BatchNormalization()(conv_2)
        conv_2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same")(conv_2_norm)
        conv_2_norm = BatchNormalization()(conv_2)

        conv_3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation="relu", padding="same")(conv_2_norm)
        conv_3_norm = BatchNormalization()(conv_3)
        conv_3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation="relu", padding="same")(conv_3_norm)
        conv_3_norm = BatchNormalization()(conv_3)
        drop_3 = Dropout(0.2)(conv_3_norm)

        conv_4 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), activation="relu", padding="same")(drop_3)
        conv_4_norm = BatchNormalization()(conv_4)
        conv_4 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), activation="relu", padding="same")(conv_4_norm)
        conv_4_norm = BatchNormalization()(conv_4)
        drop_4 = Dropout(0.2)(conv_4_norm)

        conv_5 = Conv2D(filters=128, kernel_size=(9, 9), strides=(1, 1), activation="relu", padding="same")(drop_4)
        conv_5_norm = BatchNormalization()(conv_5)
        conv_5 = Conv2D(filters=128, kernel_size=(9, 9), strides=(1, 1), activation="relu", padding="same")(conv_5_norm)
        conv_5_norm = BatchNormalization()(conv_5)
        drop_5 = Dropout(0.25)(conv_5_norm)

        conv_6 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), activation="relu", padding="same")(drop_5)
        conv_6_norm = BatchNormalization()(conv_6)
        conv_6 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), activation="relu", padding="same")(conv_6_norm)
        conv_6_norm = BatchNormalization()(conv_6)
        drop_6 = Dropout(0.2)(conv_6_norm)

        conv_7 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation="relu", padding="same")(drop_6)
        conv_7_norm = BatchNormalization()(conv_7)
        conv_7 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation="relu", padding="same")(conv_7_norm)
        conv_7_norm = BatchNormalization()(conv_7)
        drop_7 = Dropout(0.2)(conv_7_norm)

        conv_8 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same")(drop_7)
        conv_8_norm = BatchNormalization()(conv_8)
        conv_8 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same")(conv_8_norm)
        conv_8_norm = BatchNormalization()(conv_8)

        conv_9 = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), activation="relu", padding="same")(conv_8_norm)
        conv_9_norm = BatchNormalization()(conv_9)
        conv_9 = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), activation="relu", padding="same")(conv_9_norm)
        conv_9_norm = BatchNormalization()(conv_9)

        # flatten = Flatten()(drop_4)
        hidden_layer_1 = Dense(1024, activation="relu")(conv_9_norm)
        hl_1_norm = BatchNormalization()(hidden_layer_1)
        drop_1 = Dropout(0.3)(hl_1_norm)

        output_layer = Dense(3, activation="relu")(drop_1)

        model = Model(outputs=output_layer, inputs=input_layer)

        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0001, amsgrad=False)
        model.compile(loss="mean_squared_error", optimizer=opt, metrics=['acc'])
        model.summary()

    return model

if __name__ == "__main__":
    create_model(3)



