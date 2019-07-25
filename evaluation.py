from keras.models import load_model
import os
from settings import *
from PIL import Image
import numpy as np
from keras.models import model_from_json
import json
from matplotlib import pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)


def load(filename):
    json_file = os.path.join(WEIGHTS_DIR, filename)
    weight_file = os.path.join(WEIGHTS_DIR, "Model_2.h5")
    print(json_file)
    mod = load_model("D:\\Starcraft 2 AI\\Model\\CNN_mode_luke.json")
    print(weight_file)

    return mod


def load_json(filename, weightname):
    json_file = os.path.join(WEIGHTS_DIR, filename)
    weight_file = os.path.join(WEIGHTS_DIR, weightname)
    f = open(json_file, "r")
    mod = model_from_json(f.read())
    print("Model Loaded")
    mod.load_weights(weight_file)
    print("Weights Loaded")
    return mod


def plot_history(hst):
    plt.title("Loss / Mean Squared Error")
    plt.plot(hst.history["loss"], label="train")
    plt.plot(hst.history["val_loss"], label="test")
    plt.legend()
    plt.show()


def load_history():
    history_file = os.path.join(WEIGHTS_DIR, "history_model_10.json")
    return json.load(open(history_file, 'r'))


def plot_history1(his, model_id):
    if not os.path.exists(PLOT_DIR):
        os.mkdir(PLOT_DIR)

    acc_file = os.path.join(PLOT_DIR, 'history_plot_acc_'+str(model_id)+'.png')
    loss_file = os.path.join(PLOT_DIR, 'history_plot_loss_'+str(model_id)+'.png')

    history = his.history

    # summarize history for accuracy
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model '+str(model_id)+' Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(acc_file)

    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model '+str(model_id)+' Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(loss_file)


# def predict_image(model, id, batch):
#     proj_dir = "D:\\Starcraft 2 AI\\Frames\\Acid_Plant"
#     frame = "Acid_Plant_141_frame_1500.png"
#     im = Image.open(proj_dir + "\\" + frame)
#     np_im = np.array(im, dtype=np.int32)
#
#     np_arr = [np_im]
#     np.savez("to_predict.npz", x=np_arr)
#     np_arr_2 = np.load("to_predict.npz")
#     np_arr_2 = np_arr_2["x"]
#
#     prediction = model.predict(np_arr_2)
#     prediction = prediction[0]
#
#     save_prediction(prediction, id, batch)


def get_frames(map_name, replay, range_x, range_y):
    proj_dir = FRAMES_DIR + map_name
    frames = []
    for i in range(range_x, range_y+1):
        frame = map_name + "_" + str(replay) + "_frame_" + str(i) + ".png"
        # print(frame)
        im = Image.open(proj_dir + "\\" + frame)
        frame = np.array(im, dtype=np.uint8)
        frames.append(frame)

    frames = np.array(frames)

    return frames


def single_test(model_id, map_name, replay, lower_bound=0, upper_bound=0):
    model = load_json("CNN_model_" + str(model_id) + ".json", "weights_" + str(model_id) + ".h5")

    frames = get_frames(map_name, replay, lower_bound, upper_bound)

    prediction = model.predict(frames)
    prediction = prediction[0]

    save_prediction(prediction, model_id, map_name, replay, lower_bound, upper_bound)


def callback_predict(model, model_id, epoch_num):
    replay = 141
    lower_bound = 500
    upper_bound = 505
    map_name = "Acid_Plant"
    frames = get_frames(map_name, replay, lower_bound, upper_bound)

    prediction = model.predict(frames)
    prediction = prediction[0]  # todo check array

    save_prediction(prediction, model_id, map_name, replay, lower_bound=lower_bound, upper_bound=upper_bound,
                    epoch_num=epoch_num)


def save_prediction(prediction, model_id, map_name, replay, lower_bound=0, upper_bound=0, epoch_num=None):
    pred_dir = os.path.join(PREDICTION_DIR, "Model_" + str(model_id))
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    prediction = prediction.astype(np.uint8)
    img = Image.fromarray(prediction)
    if epoch_num is None:
        img.save(pred_dir + "\\prediction_" + map_name + "_" + str(replay) + "_" + str(lower_bound) + "-" +
                 str(upper_bound) + ".png")
    else:
        img.save(pred_dir + "\\prediction_" + map_name + "_" + str(replay) + "_" + str(lower_bound) + "-" +
                 str(upper_bound) + "_epoch_" + str(epoch_num) + ".png")


def checking_in_out_arrays():
    input = 0
    output = 0
    for i in range(52):
        file = "D:\\Starcraft 2 AI\\Numpy_Frames\\Acid_Plant\\Acid_Plant_" + str(i) + ".npz"
        ws = np.load(file)
        ina = ws["x"]
        oua = ws["Y"]
        print("File: ", i, " ->", len(ina), " ", len(oua))
        input += len(ina)
        output += len(oua)

    expected = 311852 - 121
    print("Total Input: ", input, " | Expected: ", expected)
    print("Total Output: ", output, " | Expected: ", expected)


if __name__ == "__main__":
    # model = load_json("CNN_model_01.json", "weights_01.h5")
    single_test(10, "Acid_Plant", 141, 1496, 1498)
    print("Evaluation Complete")
    # hst = load_history()
    # plot_history(hst)
