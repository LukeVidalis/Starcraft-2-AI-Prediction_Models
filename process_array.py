import numpy as np
import gc
from os import listdir
from os.path import isfile, join
from PIL import Image


def save_array(filename, x, Y, complete):

    np.savez(filename, x=x, Y=Y)
    np.save(filename+"_ALL", complete)


def load_array(filename):
    arr = np.load(filename)
    return arr

#np_im = None
input_frames = []
output_frames = []
all_frames = []

def process_images():


    proj_dir = "D:\\Starcraft 2 AI\\Frames\\Acid_Plant"
    frames_list = [f for f in listdir(proj_dir) if isfile(join(proj_dir, f))]

    d = 0

    for i in range(142):
        replay = "_" + str(i) + "_f"
        input = []
        output = []
        if any(replay in s for s in frames_list):
            matching = [s for s in frames_list if replay in s]
            matching.sort()
            d += 1
            print("\nReplay " + str(i))
            #while len(matching) != 0:
            for j in range(len(matching)):
                frame = "Acid_Plant_" + str(i) + "_frame_" + str(j) + ".png"
                #if (matching[j][18:19] == frame) or (matching[j][18:20] == frame) or (matching[j][18:121] == frame):
                im = Image.open(proj_dir + "\\" + frame)
                print(frame)
                #global np_im
                #print(d)
                #d += 1
                np_im = None
                np_im = np.array(im)
                input.append(np_im)
                output.append(np_im)
                all_frames.append(np_im)
                #matching.pop(j)
                #gc.collect()
            #print("inputting in arrays")
            input.pop(len(input)-1)
            output.pop(0)
            input_frames + input
            output_frames + output
            print(str(i) + " finished  -> Next Replay")
        gc.collect()

    save_array("Acid_Plant", input_frames, output_frames, all_frames)
    print("\nAll Frames: " + str(len(all_frames)))


if __name__ == "__main__":
    process_images()
