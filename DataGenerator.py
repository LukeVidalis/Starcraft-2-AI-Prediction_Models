import numpy as np
import keras
from os import listdir
from os.path import isfile, join
from settings import *


class DataGenerator(keras.utils.Sequence):
    # 'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(128, 128), n_channels=3,
                 shuffle=True):
        # 'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.n_channels = n_channels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        file_list = [f for f in listdir(DATA_DIR) if isfile(join(DATA_DIR, f))]

        # Generate data
        for file in file_list:

            # Store sample
            X = file['x']

            # Store class
            y = file['Y']

        yield X, y
