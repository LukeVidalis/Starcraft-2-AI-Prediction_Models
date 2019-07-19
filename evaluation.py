from keras.models import load_model
import os
from settings import *
from PIL import Image
import numpy as np
from keras.models import model_from_json
from matplotlib import pyplot
import sys
np.set_printoptions(threshold=sys.maxsize)


def load(filename):
    json_file = os.path.join(WEIGHTS_DIR, filename)
    weight_file = os.path.join(WEIGHTS_DIR, "Model_2.h5")
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
    " code here "


def plot_history(hst):
    pyplot.title("Loss / Mean Squared Error")
    pyplot.plot(hst.history["loss"], label="train")
    pyplot.plot(hst.history["val_loss"], label="test")
    pyplot.legend()
    pyplot.show()


def predict_image(model, id, batch):
    proj_dir = "D:\\Starcraft 2 AI\\Frames\\Acid_Plant"
    frame = "Acid_Plant_141_frame_1500.png"
    im = Image.open(proj_dir + "\\" + frame)
    np_im = np.array(im, dtype=np.int32)

    np_arr = []
    np_arr.append(np_im)
    np.savez("to_predict.npz", x=np_arr)
    np_arr_2 = np.load("to_predict.npz")
    np_arr_2 = np_arr_2["x"]

    prediction = model.predict(np_arr_2)
    prediction = prediction[0]  # np.resize(out, (128, 128, 3))

    save_prediction(prediction, id, batch)


def save_prediction(prediction, id, batch):
    pred_dir = TRAIN_PREDICTION_DIR + "\\Model_" + str(id)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    prediction = (prediction * 255).astype(np.uint8)
    img = Image.fromarray(prediction)
    img.save(pred_dir + "prediction_" + str(batch) + ".png")

    # ByteBuffer buffer = ByteBuffer.wrap(os.toByteArray());
    # buffer.rewind();
    # bitmap_tmp.copyPixelsFromBuffer(buffer);
    # img = image.array_to_img(out)
    # img.save("predict_2.png")


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
    model = load_json("CNN_model_9.json", "weights_9.h5")
    predict_image(model)
    print("Evaluation Complete")
