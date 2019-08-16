from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Input, BatchNormalization, Dropout
from keras.layers import ConvLSTM2D, TimeDistributed
from keras.optimizers import Adam

# Parameters
img_width = 128
img_height = 128
rgb = 3

# Method to create and compile the CNN and ConvLSTM models
def create_model(model_index):

    print("Creating Model")
    if model_index == 1:
        input_layer = Input(shape=(128, 128, 3))
        in_norm = BatchNormalization()(input_layer)

        conv_1 = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), activation="relu", padding="same")(in_norm)
        conv_1_norm = BatchNormalization()(conv_1)

        conv_2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same")(conv_1_norm)
        conv_2_norm = BatchNormalization()(conv_2)

        conv_3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation="relu", padding="same")(conv_2_norm)
        conv_3_norm = BatchNormalization()(conv_3)
        drop_3 = Dropout(0.2)(conv_3_norm)

        conv_4 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), activation="relu", padding="same")(drop_3)
        conv_4_norm = BatchNormalization()(conv_4)
        drop_4 = Dropout(0.2)(conv_4_norm)

        conv_6 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), activation="relu", padding="same")(drop_4)
        conv_6_norm = BatchNormalization()(conv_6)
        drop_6 = Dropout(0.2)(conv_6_norm)

        conv_7 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation="relu", padding="same")(drop_6)
        conv_7_norm = BatchNormalization()(conv_7)
        drop_7 = Dropout(0.2)(conv_7_norm)

        conv_8 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same")(drop_7)
        conv_8_norm = BatchNormalization()(conv_8)

        conv_9 = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), activation="relu", padding="same")(conv_8_norm)
        conv_9_norm = BatchNormalization()(conv_9)

        output_layer = Dense(3, activation="relu")(conv_9_norm)

        model = Model(outputs=output_layer, inputs=input_layer)

        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0001, amsgrad=False)
        model.compile(loss="mean_squared_error", optimizer=opt, metrics=['acc'])
        model.summary()
    elif model_index == 2:
        cnn = Sequential()

        cnn.add(Conv2D(filters=16, input_shape=(128, 128, 3), kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"))
        cnn.add(BatchNormalization())

        cnn.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation="relu", padding="same"))
        cnn.add(BatchNormalization())

        cnn.add(Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), activation="relu", padding="same"))
        cnn.add(BatchNormalization())

        cnn.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation="relu", padding="same"))
        cnn.add(BatchNormalization())

        cnn.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"))
        cnn.add(BatchNormalization())

        rnn = Sequential()

        rnn.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same", return_sequences=True))
        rnn.add(BatchNormalization())

        rnn.add(ConvLSTM2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation="relu",padding="same", return_sequences=False))
        rnn.add(BatchNormalization())

        output = Sequential()
        output.add(Dense(3, activation="relu"))

        main_input = Input((3, 128, 128, 3))

        merge = TimeDistributed(cnn)(main_input)
        merge = rnn(merge)
        merge = output(merge)

        model = Model(inputs=main_input, outputs=merge)
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0001, amsgrad=False)
        model.compile(loss="mean_squared_error", optimizer=opt, metrics=['acc'])
        model.summary()

    return model


if __name__ == "__main__":
    model = create_model(1)

