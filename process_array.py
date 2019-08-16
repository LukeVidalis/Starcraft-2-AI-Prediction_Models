import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image


# Method to save the numpy arrays in a file
def save_array(filename, x, Y):

    np.savez(filename, x=x, Y=Y)


# Method to load numpy files
def load_array(filename):
    arr = np.load(filename)
    return arr


# Method to proccess the data for the ConvLSTM model
def process_images_RNN(r_map, type, lower_bound, upper_bound):
    input_frames = []
    output_frames = []

    proj_dir = "D:\\Starcraft 2 AI\\Frames\\"+r_map
    frames_list = [f for f in listdir(proj_dir) if isfile(join(proj_dir, f))]

    for i in range(0, 7):
        replay = "_" + str(i) + "_f"

        if any(replay in s for s in frames_list):
            matching = [s for s in frames_list if replay in s]
            print("\nReplay " + str(i))
            for j in range(len(matching)):
                if j + 7 < len(matching):
                    seq = []
                    for k in range(j, j+6, 2):
                        frame = r_map + "_" + str(i) + "_frame_" + str(k) + ".png"
                        im = Image.open(proj_dir + "\\" + frame)
                        np_im = np.array(im)
                        seq.append(np_im)
                    input_frames.append(seq)
                    frame = r_map + "_" + str(i) + "_frame_" + str(j+7) + ".png"
                    im = Image.open(proj_dir + "\\" + frame)
                    np_im = np.array(im)
                    output_frames.append(np_im)

    save_name = r_map + "_" + type + "_RNN"
    save_array(save_name, input_frames, output_frames)


# Method to proccess the data for the CNN model
def process_images_CNN(r_map, type, lower_bound, upper_bound):
    input_frames = []
    output_frames = []

    proj_dir = "D:\\Starcraft 2 AI\\Frames\\"+r_map
    frames_list = [f for f in listdir(proj_dir) if isfile(join(proj_dir, f))]

    for i in range(lower_bound, upper_bound):
        replay = "_" + str(i) + "_f"
        input = []
        output = []

        if any(replay in s for s in frames_list):
            matching = [s for s in frames_list if replay in s]
            print("\nReplay " + str(i))

            for j in range(len(matching)):
                frame = r_map + "_" + str(i) + "_frame_" + str(j) + ".png"
                im = Image.open(proj_dir + "\\" + frame)
                np_im = np.array(im)
                input.append(np_im)

            for k in range(len(input)):
                if k + 2 <= len(input)-1:
                    output.append(input[k+2])

            input.pop(len(input) - 1)
            input.pop(len(input) - 1)
            input_frames += input
            output_frames += output

    save_name = r_map + "_" + type + "_CNN"
    save_array(save_name, input_frames, output_frames)


# Method to check the sizes of the arrays
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
    process_images_RNN("Port_Aleksandrer", "Test", 0, 11)
