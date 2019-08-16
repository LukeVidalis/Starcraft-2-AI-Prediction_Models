import os
import time
import math
from datetime import datetime
from threading import Timer
from process_array import *
from CNN import create_model
import pandas as pd
from settings import *
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from callbacks import CallbackPred
import tensorflow as tf
from keras import backend as k
from evaluation import plot_history1

# Parameters
model_id = 13
epochs_num = 100
batch_size = 32
val_split = 0.2

# Paths
json_file = os.path.join(WEIGHTS_DIR, 'CNN_model_'+str(model_id)+'.json')
weights_file = os.path.join(WEIGHTS_DIR, 'weights_'+str(model_id)+'.h5')
history_file = os.path.join(WEIGHTS_DIR, 'history_model_' + str(model_id) + '.json')
dataset = os.path.join(DATA_DIR, "Acid_Plant_0.npz")
model_checkpoint = os.path.join(WEIGHTS_DIR, 'model_' + str(model_id) + '_checkpoint.h5')


# Set up GPU parameters for training
def gpu_setup():
    config = tf.ConfigProto()
    # Allocate memory as needed. No pre-allocation.
    config.gpu_options.allow_growth = True
    # Allow memory allocation up to this percentage.
    config.gpu_options.per_process_gpu_memory_fraction = 0.65
    # Create session with the above options.
    k.tensorflow_backend.set_session(tf.Session(config=config))


# Load the training and testing data and split them with the option to shuffle
def load_files(data_path, shuffle=False):
    print("Getting Data")
    data = load_array(data_path)
    x_val = data['x']
    y_val = data['Y']

    if shuffle:
        c = list(zip(x_val, y_val))
        np.random.shuffle(c)
        x_val, y_val = zip(*c)

    return x_val, y_val


# Generator method for  fit_generator
def data_generator(x_data, y_data, bs):
    iter_x = iter(x_data)
    iter_y = iter(y_data)
    # loop indefinitely
    while True:
        # initialize our batches of input and output
        images_x = []
        images_y = []

        # keep looping until we reach our batch size
        while len(images_x) < bs:
            try:
                images_x.append(next(iter_x))
                images_y.append(next(iter_y))
            except StopIteration:
                iter_x = iter(x_data)
                iter_y = iter(y_data)

        images_x = np.array(images_x)
        images_y = np.array(images_y)

        yield (images_x, images_y)


# Save model structure, weights and history object
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


# Callback for learning rate adjustment
def lr_schedule():
    return lambda epoch: 0.001 if epoch < 75 else 0.0001


# Training Method
def train_model(model, x, Y):
    print("Training Model")
    split_id = math.floor(len(x)*(1-val_split))
    training_generator = data_generator(x[:split_id], Y[:split_id], batch_size)
    testing_generator = data_generator(x[split_id:], Y[:split_id], batch_size)
    steps_per_epoch = math.ceil(len(x[:split_id])/batch_size)
    val_steps_per_epoch = math.ceil(len(x[split_id:])/batch_size)

    callbacks = [ModelCheckpoint(filepath=model_checkpoint, monitor='val_loss',
                                 verbose=1, save_best_only=True), CallbackPred(period=10, model_id=model_id)]

    print("Epochs: "+str(epochs_num)+"\nBatch Size: "+str(batch_size))
    start = time.time()

    hst = model.fit_generator(training_generator, steps_per_epoch=steps_per_epoch, epochs=epochs_num, verbose=2,
                              callbacks=callbacks, validation_data=testing_generator,
                              validation_steps=val_steps_per_epoch, class_weight=None,
                              max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)

    end = time.time()
    print("Time Elapsed: "+str(end-start))
    plot_history1(hst, model_id)
    return hst, model


# Method to schedule training
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
    t = Timer(secs, actions_generator)
    hours = int(secs / 3600)
    mins = (secs - (3600*hours))/60
    print("Timer started: "+str(hours)+" hours and "+str(mins)+" minutes remaining.")
    t.start()


# Goes through the training process
def actions_generator():
    gpu_setup()
    seq_model = create_model(1)
    x, Y = load_files(dataset)
    history, seq_model = train_model(seq_model, x, Y)
    # predict_image(seq_model, model_id)
    save_model(seq_model, history)
    print(seq_model.summary())


if __name__ == "__main__":
    usr_input = input("Do you want to set up a scheduled run? (y/n)")
    if usr_input == "y" or usr_input == "Y" or usr_input == "yes":
        schedule()
    elif usr_input == "n" or usr_input == "N" or usr_input == "no":
        actions_generator()
    else:
        print("Input not recognized.")
