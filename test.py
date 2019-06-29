import os


# Function to rename multiple files
def main():
    i = 0
    dir = "C:\\Users\\Lucas\\Desktop\\Replays\\Acid_Plant_(139)\\"
    for filename in os.listdir(dir):
        dst = "Acid_Plant_" + str(i) + ".SC2Replay"
        src = dir + filename
        dst = dir + dst
        os.rename(src, dst)
        i += 1


# Driver Code
if __name__ == '__main__':
    main()
