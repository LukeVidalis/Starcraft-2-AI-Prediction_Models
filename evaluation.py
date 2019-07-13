from keras.models import load_model
import os
from settings import *
from PIL import Image
import numpy
from CNN import create_model
from keras.preprocessing.image import img_to_array, load_img


def load(filename):
    json_file = os.path.join(WEIGHTS_DIR, filename)
    weight_file = os.path.join(WEIGHTS_DIR, "Model2.h5")
    print(json_file)
    mod = load_model(json_file)
    print(weight_file)
    mod.load_weights(weight_file)
    return mod


def evaluate(mod):
    # im = Image.open("./Frames/Acid_Plant/Acid_Plant_80_frame_3117.png")
    # np_im = numpy.array(im)
    # x = img_to_array(im)
    img = load_img('./Frames/Acid_Plant/Acid_Plant_80_frame_3117.png')  # this is a PIL image
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    # print(x)
    x = x.reshape((128, 128, 3))

    Image.fromarray(x.astype('uint8')).save("input.png")

    out = mod.predict(x)
    print(numpy.array_equal(x, out))

    # out = out.astype(int)
    out = out.reshape((1,) + out.shape)

    # print(out)
    Image.fromarray(out.astype('uint8')).save("prediction.png")


if __name__ == "__main__":
    model = load("CNN_model2.json")
    # model = create_model()
    print("Model Created")
    model.load_weights("./Model/model1.h5")
    print("Weights Loaded")

    evaluate(model)
