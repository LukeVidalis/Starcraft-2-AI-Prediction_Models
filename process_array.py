import numpy as np


def save_array(filename, x, Y, complete):

    np.savez(filename, x=x, Y=Y)
    np.save(filename+"_ALL", complete)


def load_array(filename):
    arr = np.load(filename)
    return arr

# if __name__ == "__main__":
#     save_array([], [], "test")
