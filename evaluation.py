from keras.models import load_model
import os
from settings import *
from PIL import Image
import numpy as np
from CNN import create_model
import sys
np.set_printoptions(threshold=sys.maxsize)
from keras.preprocessing.image import save_img


def load(filename):
    json_file = os.path.join(WEIGHTS_DIR, filename)
    weight_file = os.path.join(WEIGHTS_DIR, "Model1.h5")
    print(json_file)
    mod = load_model("D:\\Starcraft 2 AI\\Model\\CNN_mode_luke.json")
    print(weight_file)
    #mod.load_weights("D:\\Starcraft 2 AI\\Model\\model1_luke.json")
    return mod


def evaluate(model):
    proj_dir = "D:\\Starcraft 2 AI\\Frames\\Acid_Plant"
    frame = "Acid_Plant_0_frame_1500.png"
    im = Image.open(proj_dir + "\\" + frame)
    np_im = np.array(im, dtype=np.int32)
    #print(np_im)
    #np_im = np.resize(np_im, (1, 128, 128, 3))
    #print("--------------\n", np_im)
    np_arr = []
    np_arr.append(np_im)
    np.savez("test.npz", x=np_arr)
    np_arr_2 = np.load("test.npz")
    np_arr_2 = np_arr_2["x"]
    #np_arr.append(np_im)
    out = model.predict(np_arr_2)
    #out = out.astype(int)
    out = np.resize(out, (128, 128, 3))
    save_img("prediction.png", out)
    #Image.fromarray(out.astype('uint8')).save("prediction.png")
    print(out)


if __name__ == "__main__":
    #mod = load_model("D:\\Starcraft 2 AI\\Model\\CNN_model_luke.json")
    model = create_model()
    print("Model Created")
    model.load_weights("D:\\Starcraft 2 AI\\Model\\model1_luke.h5")
    print("Weights Loaded")
    #model = None
    evaluate(model)
