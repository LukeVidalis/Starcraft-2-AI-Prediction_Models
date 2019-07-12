from keras.models import load_model
import os
from settings import *
from PIL import Image
import numpy
from CNN import create_model


def load(filename):
    json_file = os.path.join(WEIGHTS_DIR, filename)
    weight_file = os.path.join(WEIGHTS_DIR, "Model1.h5")
    print(json_file)
    mod = load_model(json_file)
    print(weight_file)
    mod.load_weights(weight_file)
    return mod


def evaluate(model):
    im = Image.open("./Frames/Acid_Plant/Acid_Plant_80_frame_3117.png")
    np_im = numpy.array(im)
    out = model.predict(np_im)
    Image.fromarray(out.astype('uint8')).save("prediction.png")
    print(out)


if __name__ == "__main__":
    # model = load("CNN_model.json")
    model = create_model()
    print("Model Created")
    model.load_weights("./Model/model1.h5")
    print("Weights Loaded")

    evaluate(model)
