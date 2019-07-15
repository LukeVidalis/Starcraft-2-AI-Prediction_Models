import numpy as np
import gc
from os import listdir
from os.path import isfile, join
from PIL import Image


def save_array(filename, x, Y):

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
            #while len(matching) != 0:
            for j in range(len(matching)):
                frame = "Acid_Plant_" + str(i) + "_frame_" + str(j) + ".png"
                #if (matching[j][18:19] == frame) or (matching[j][18:20] == frame) or (matching[j][18:121] == frame):
                im = Image.open(proj_dir + "\\" + frame)
                #print(frame)
                np_im = np.array(im)
                input.append(np_im)
                output.append(np_im)
                #all_frames.append(np_im)
            input.pop(len(input)-1)
            output.pop(0)
            input_frames += input
            output_frames += output
            print(len(input_frames))
            print(len(output_frames))
            if len(input_frames) >= 5884 and len(output_frames) >= 5884:
                print("Creating File")
                file_name = save_dir + "\\Acid_Plant_" + str(file_number)
                in_arr = input_frames[:5884]
                out_arr = output_frames[:5884]
                #print(len(in_arr))
                save_array(file_name, in_arr, out_arr)
                input_frames = input_frames[5885:]
                output_frames = output_frames[5885:]
                #all_frames = all_frames[5884:]
                print(len(input_frames))
                print(len(output_frames))
                file_number += 1

    file_name = save_dir + "\\Acid_Plant_" + str(file_number)
    save_array(file_name, input_frames, output_frames)
    print("\ninput_frames: " + str(len(input_frames)))
    print("\noutput_frames: " + str(len(output_frames)))
    print("\nAll Frames: " + str(len(all_frames)))
            #print("\nAll Frames: " + str(len(all_frames)))
            #print(str(i) + " finished  -> Next Replay")
        #gc.collect()
    #
    # for i in range(53):
    #     file_name = save_dir + "\\Acid_Plant_" + str(i)
    #     in_arr = input_frames[:5884]
    #     out_arr = output_frames[:5884]
    #     print(len(in_arr))
    #     save_array(file_name, input_frames, output_frames)
    #     for j in range(5884):
    #         input_frames.pop(j)
    #         output_frames.pop(j)
    #         all_frames.pop(j)




        # if len(all_frames) == 5884:
        #     print("\nAll Frames: " + str(len(all_frames)))
        #     file_name = save_dir + "\\Acid_Plant_" + str(file_number)
        #     save_array(file_name, input_frames, output_frames)
        #     input_frames = []
        #     output_frames = []
        #     all_frames = []

    #save_array("Acid_Plant", input_frames, output_frames, all_frames)
    #print("\nAll Frames: " + str(len(all_frames)))


if __name__ == "__main__":
    process_images()
