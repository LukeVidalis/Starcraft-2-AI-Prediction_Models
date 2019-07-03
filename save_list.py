import csv
from PIL import Image

def main():
    pixels = [[['1a', '2a', '3a'], ['4a', '5a', '6a'], ['7a', '8a', '9a']],
              [['1b', '2b', '3b'], ['4b', '5b', '6b']],
              [['1c', '2c', '3c'], ['4c', '5c', '6c']]
              ]

    # with open("test.csv", 'w+', newline='') as myfile:
    #     csvWriter = csv.writer(myfile, delimiter=',')
    #     csvWriter.writerows(pixels)
    i = 0
    with open('test_images.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            print(row)
            Image.fromarray(row.astype('uint8')).save(str(i)+'.png')
            i = i+1


if __name__ == '__main__':
    main()
