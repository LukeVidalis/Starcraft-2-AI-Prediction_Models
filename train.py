import os
import time
import datetime
from keras.models import load_model
from process_array import *
from settings import *
from CNN import create_model


# Parameters
model_id = 7
img_width = 128
img_height = 128
rgb = 3
epochs_num = 100
batch_size = 1


# Paths
json_file = os.path.join(WEIGHTS_DIR, 'CNN_model_'+str(model_id)+'.json')


def load_files():
    print("Getting Data")
    data = load_array("Acid_Plant10.npz")
    return data['x'], data['Y']


def save_model(model):
    print("Saving Model")
    json_string = seq_model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)
    model.save_weights("model"+str(model_id)+".h5")


def get_model():
    return load_model(json_file)


def train_model(model, x, Y):
    print("Training Model")
    print("Epochs: "+str(epochs_num)+"\nBatch Size: "+str(batch_size))
    start = time.time()
    hst = model.fit(x=x, y=Y, batch_size=batch_size, epochs=epochs_num, verbose=2, callbacks=None,
                    validation_split=0.2, validation_data=None, shuffle=True, class_weight=None,
                    sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
    end = time.time()
    print("Time Elapsed: "+str(end-start))
    return hst, model

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
    x, Y = load_files()
    seq_model = create_model()
    history, seq_model = train_model(seq_model, x, Y)
    save_model(seq_model)
