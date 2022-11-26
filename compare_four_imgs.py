
import numpy as np
import cv2 as cv
import matplotlib.pyplot as pltfrom 
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import math

from main import fft_2d


def mean_filter(img):
    mean_img = cv.blur(img, (3,3))
    return mean_img

def median_filter(img, filter_kernel):
    temp = []
    
    indexer = filter_kernel // 2
    data_final = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    for i in range(len(img) - indexer):
        for j in range(len(img[0])- indexer):

            for z in range(filter_kernel-1):
                if i + z - indexer < 0 or i + z - indexer > len(img) + 1:
                    for c in range(filter_kernel):
                        temp.append(0)

                else:
                    if j + z  - indexer < 0 or j + z - indexer > len(img[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_kernel):
                            temp.append(img[i + z - indexer][j + k - indexer])
            
            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []

    return data_final

def image_resize(img):
    return cv.resize(img, (400, 400))

def read_raw_img(img):

    fd = open(img, 'rb')
    rows = 512
    cols = 512
    f = np.fromfile(fd, dtype = np.uint8, count = rows * cols)
    raw_img = f.reshape((rows, cols)) #notice row, column format
    fd.close()

    return raw_img


    
def main():
    a_img = read_raw_img('pirate_a.raw')
    b_img = read_raw_img('pirate_b.raw')

    mean_a = mean_filter(a_img)
    mean_a = image_resize(mean_a)

    mean_b = mean_filter(b_img)
    mean_b = image_resize(mean_b)

    median_a = median_filter(a_img, 3)
    median_a = image_resize(median_a)

    median_b = median_filter(b_img, 3)
    median_b = image_resize(median_b)

    # cv.imshow("A - mean filter", mean_a)
    # cv.imshow("B - mean filter", mean_b)
    # cv.imshow("A - median filter", median_a)
    # cv.imshow("B - median filter", median_b)
    
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()


# median a is better
