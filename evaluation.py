from keras.models import load_model
import os
from settings import *


def load(filename):
    json_file = os.path.join(WEIGHTS_DIR, filename)
    weight_file = os.path.join(WEIGHTS_DIR, "Model1.h5")
    mod =   load_model(json_file)
    mod.load_weights(weight_file)
    return mod

def evaluate(model):
    



if __name__ == "__main__":
    model = load("CNN_model.json")
    evaluate(model)
