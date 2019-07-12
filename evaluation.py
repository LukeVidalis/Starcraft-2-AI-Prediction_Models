from keras.models import load_model
from settings import *
import os

json_file_model = os.path.join(WEIGHTS_DIR, 'CNN_model.json')
json_file_weights = os.path.join(WEIGHTS_DIR, 'model2.h5')
seq_model = None
model_weights = None

def load():
    seq_model = load_model(json_file_model)



if __name__ == "__main__":
    load()
