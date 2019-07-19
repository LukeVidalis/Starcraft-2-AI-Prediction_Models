import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image


def save_array(filename, x, Y):

    np.savez(filename, x=x, Y=Y)


def load_array(filename):
    arr = np.load(filename)
    return arr


def process_images():
    input_frames = []
    output_frames = []
    all_frames = []

    proj_dir = "D:\\Starcraft 2 AI\\Frames\\Acid_Plant"
    save_dir = "D:\\Starcraft 2 AI\\Numpy_Frames\\Acid_Plant"
    frames_list = [f for f in listdir(proj_dir) if isfile(join(proj_dir, f))]

    file_number = 0

    for i in range(150):
        replay = "_" + str(i) + "_f"
        input = []
        output = []

        if any(replay in s for s in frames_list):
            matching = [s for s in frames_list if replay in s]
            print("\nReplay " + str(i))

            for j in range(len(matching)):
                frame = "Acid_Plant_" + str(i) + "_frame_" + str(j) + ".png"
                im = Image.open(proj_dir + "\\" + frame)
                np_im = np.array(im)
                input.append(np_im)
                output.append(np_im)

            input.pop(len(input)-1)
            output.pop(0)
            input_frames += input
            output_frames += output

            if len(input_frames) >= 5884 and len(output_frames) >= 5884:
                print("Creating Batch")
                file_name = save_dir + "\\Acid_Plant_" + str(file_number)
                in_arr = input_frames[:5884]
                out_arr = output_frames[:5884]
                save_array(file_name, in_arr, out_arr)
                input_frames = input_frames[5885:]
                output_frames = output_frames[5885:]
                print(len(input_frames))
                print(len(output_frames))
                file_number += 1

    file_name = save_dir + "\\Acid_Plant_" + str(file_number)
    save_array(file_name, input_frames, output_frames)
    print("\ninput_frames: " + str(len(input_frames)))
    print("\noutput_frames: " + str(len(output_frames)))
    print("\nAll Frames: " + str(len(all_frames)))


if __name__ == "__main__":
    process_images()
