import os
import time
from datetime import datetime
from threading import Timer
from keras.models import load_model
from process_array import *
from CNN import create_model
import pandas as pd
from settings import *
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from evaluation import predict_image
# Parameters
model_id = 7
epochs_num = 100
batch_size = 1
val_split = 0.2

# Paths
json_file = os.path.join(WEIGHTS_DIR, 'CNN_model_'+str(model_id)+'.json')
weights_file = os.path.join(WEIGHTS_DIR, 'weights_'+str(model_id)+'.h5')
history_file = os.path.join(WEIGHTS_DIR, 'history_model_' + str(model_id) + '.json')

# dataset = "Acid_Plant10.npz"


def load_files(dataset):
    print("Getting Data")
    data = load_array(dataset)
    return data['x'], data['Y']


def generator(x, Y):
    # Create empty arrays to contain batch of features and labels
    file_list = [f for f in listdir(DATA_DIR) if isfile(join(DATA_DIR, f))]

    # Generate data
    for file in file_list:
        arr = load_array(file)
        # Store sample
        x = arr['x']

        # Store class
        Y = arr['Y']

    yield x, Y


def save_model(model, hst):
    if not os.path.exists(WEIGHTS_DIR):
        os.mkdir(WEIGHTS_DIR)

    print("Saving Model")
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)
    model.save_weights(weights_file)

    hist_df = pd.DataFrame(hst.history)
    with open(history_file, mode='w') as f:
        hist_df.to_json(f)


def lr_schedule():
    return lambda epoch: 0.001 if epoch < 75 else 0.0001


def train_model(model, x, Y):
    print("Training Model")
    callbacks = [LearningRateScheduler(lr_schedule()), ModelCheckpoint(filepath=WEIGHTS_DIR, monitor='val_loss',
                                                                       save_best_only=True)]

    print("Epochs: "+str(epochs_num)+"\nBatch Size: "+str(batch_size))
    start = time.time()

    hst = model.fit(x=x, y=Y, batch_size=batch_size, epochs=epochs_num, verbose=2, callbacks=callbacks,
                    validation_split=val_split, validation_data=None, shuffle=True, class_weight=None,
                    sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)

    # hst = model.fit_generator(generator, steps_per_epoch=92, epochs=epochs_num, verbose=2, callbacks=callbacks,
    #                           validation_data=None, validation_steps=None, validation_freq=1, class_weight=None,
    #                           max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)

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


def schedule():
    now = datetime.today()
    while True:
        try:
            target = datetime.strptime(input('Specify time in HHMM format: '), "%H%M")
            break
        except ValueError:
            print("Please write the time in HHMM format.")
    total_time = target - now
    secs = total_time.seconds+1
    t = Timer(secs, actions)
    hours = int(secs / 3600)
    mins = (secs - (3600*hours))/60
    print("Timer started: "+str(hours)+" hours and "+str(mins)+" minutes remaining.")
    t.start()


def actions():
    batch_num = 0
    file_list = [f for f in listdir(DATA_DIR) if isfile(join(DATA_DIR, f))]
    seq_model = create_model(1)
    history = None
    for file in file_list:
        print(batch_num)
        batch_num += 1
        x, Y = load_files(file)
        history, seq_model = train_model(seq_model, x, Y)
        predict_image(seq_model)
    save_model(seq_model, history)


if __name__ == "__main__":
    usr_input = input("Do you want to set up a scheduled run? (y/n)")
    if usr_input == "y" or usr_input == "Y" or usr_input == "yes":
        schedule()
    elif usr_input == "n" or usr_input == "N" or usr_input == "no":
        actions()
    else:
        print("Input not recognized.")
