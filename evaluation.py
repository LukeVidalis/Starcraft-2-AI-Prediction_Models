from keras.models import load_model
import os
from settings import *
from PIL import Image
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array, load_img, save_img
import sys
np.set_printoptions(threshold=sys.maxsize)


def load(filename):
    json_file = os.path.join(WEIGHTS_DIR, filename)
    weight_file = os.path.join(WEIGHTS_DIR, "Model2.h5")
    print(json_file)
    mod = load_model("D:\\Starcraft 2 AI\\Model\\CNN_mode_luke.json")
    print(weight_file)
    # mod.load_weights("D:\\Starcraft 2 AI\\Model\\model1_luke.json")
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


def evaluate(model):
    proj_dir = "D:\\Starcraft 2 AI\\Frames\\Acid_Plant"
    frame = "Acid_Plant_0_frame_1500.png"
    im = Image.open(proj_dir + "\\" + frame)
    np_im = np.array(im, dtype=np.int32)
    np_arr = []
    np_arr.append(np_im)
    np.savez("test.npz", x=np_arr)
    np_arr_2 = np.load("test.npz")
    np_arr_2 = np_arr_2["x"]
    out = model.predict(np_arr_2)
    out = np.resize(out, (128, 128, 3))
    save_img(frame+"_pred.png", out)
    print(out)


if __name__ == "__main__":
    model = load_json("CNN_model2.json", "Model2.h5")
    evaluate(model)
    print("Evaluation Complete")
