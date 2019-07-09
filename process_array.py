import numpy as np
from os import listdir
from os.path import isfile, join
import PIL as pil


def save_array(filename, x, Y, complete):

    np.savez(filename, x=x, Y=Y)
    np.save(filename+"_ALL", complete)


def load_array(filename):
    arr = np.load(filename)
    return arr

def process_images():
    input_frames = []
    output_frames = []
    all_frames = []

    proj_dir = "D:\\Starcraft 2 AI\\Frames\\Acid_Plant"
    frames_list = [f for f in listdir(proj_dir) if isfile(join(proj_dir, f))]

    for i in range(142):
        replay = str(i) + "_f"
        input = []
        output = []
        if any(replay in s for s in frames_list):
            matching = [s for s in frames_list if replay in s]
            matching.sort()
            all_frames += matching
            input = matching
            input.pop(len(input)-1)
            output = matching
            matching.pop(0)
            input_frames += input
            output_frames += output

    save_array("Acid_Plant", input_frames, output_frames, all_frames)


if __name__ == "__main__":
    process_images()
