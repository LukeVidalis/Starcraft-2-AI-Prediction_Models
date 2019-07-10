import numpy as np
import gc
from os import listdir
from os.path import isfile, join
from PIL import Image


def save_array(filename, x, Y, complete):

    np.savez(filename, x=x, Y=Y)
    #np.save(filename+"_ALL", complete)


def load_array(filename):
    arr = np.load(filename)
    return arr


def process_images():
    input_frames = []
    output_frames = []
    all_frames = []

    proj_dir = "D:\\Starcraft 2 AI\\Frames\\Acid_Plant"
    frames_list = [f for f in listdir(proj_dir) if isfile(join(proj_dir, f))]

    for i in range(50):
        replay = "_" + str(i) + "_f"
        input = []
        output = []
        if any(replay in s for s in frames_list):
            matching = [s for s in frames_list if replay in s]
            print("\nReplay " + str(i))
            #while len(matching) != 0:
            for j in range(len(matching)):
                frame = "Acid_Plant_" + str(i) + "_frame_" + str(j) + ".png"
                #if (matching[j][18:19] == frame) or (matching[j][18:20] == frame) or (matching[j][18:121] == frame):
                im = Image.open(proj_dir + "\\" + frame)
                print(frame)
                np_im = np.array(im)
                input.append(np_im)
                output.append(np_im)
                #all_frames.append(np_im)
            input.pop(len(input)-1)
            output.pop(0)
            input_frames += input
            output_frames += output
            print(str(i) + " finished  -> Next Replay")
        gc.collect()

    save_array("Acid_Plant", input_frames, output_frames, all_frames)
    print("\nAll Frames: " + str(len(all_frames)))


if __name__ == "__main__":
    process_images()
