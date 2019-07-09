import os
import sys

# Function to rename multiple files
def main():
    main_dir = "D:\\Starcraft 2 AI\\New Replays\\"
    x = 0
    for r in os.listdir(main_dir):
        i = 0
        replay_dir = main_dir+r+"\\"
        for filename in os.listdir(replay_dir):
            dst = r+"_" + str(i) + ".SC2Replay"
            src = replay_dir + filename
            dst = replay_dir + dst
            os.rename(src, dst)
            i += 1
            x += 1


def sorting():
    a = ["1_00001", "1_00002", "1_00000", "1_00003", "1_00006", "2_00001", "2_00000", "2_00004", "1_34345", "1_00399",
         "2_00002", "3_00001", "2_00032", "3_00000", "3_02323"]
    print(a)
    a.sort()
    print(a)
    x = sorted(a, key=lambda item: (int(item.partition(' ')[0])
                                    if item[0].isdigit() else float('inf'), item))
    print(x)


def memory():
    # check if its 64 bit version of python
    print("%x" % sys.maxsize, sys.maxsize > 2**32)


if __name__ == '__main__':
    # main()
    sorting()
    memory()
