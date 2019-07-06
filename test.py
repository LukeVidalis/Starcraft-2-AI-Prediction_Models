import os


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


# Driver Code
if __name__ == '__main__':
    main()
