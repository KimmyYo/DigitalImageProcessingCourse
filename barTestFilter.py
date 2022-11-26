import numpy as np
import cv2 as cv
import matplotlib.pyplot as pltfrom 
from matplotlib import pyplot as plt

def open__img():
    
    input_img = cv.imread("BarTest.tif")
    return input_img


def main():
    bartest_img = open__img()

    # 3x3 mean
    three_mean_img = cv.blur(bartest_img.copy(), (3, 3))

    # 7x7 mean
    seven_mean_img = cv.blur(bartest_img.copy(), (7, 7))


    # 3x3 median
    three_median_img = cv.medianBlur(bartest_img.copy(), 3)


    #7x7 median
    seven_median_img = cv.medianBlur(bartest_img.copy(), 7)

    plt.subplot(221),plt.imshow(three_mean_img, cmap = 'gray')
    plt.title('3x3 mean filter'), plt.xticks([]), plt.yticks([])

    plt.subplot(222),plt.imshow(seven_mean_img, cmap = 'gray')
    plt.title('7x7 mean filter'), plt.xticks([]), plt.yticks([])

    plt.subplot(223),plt.imshow(three_median_img, cmap = 'gray')
    plt.title('3x3 median filter'), plt.xticks([]), plt.yticks([])

    plt.subplot(224),plt.imshow(three_median_img, cmap = 'gray')
    plt.title('7x7 median filter'), plt.xticks([]), plt.yticks([])

    plt.show()
   

if __name__ == "__main__":
    main()