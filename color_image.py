import numpy as np
import cv2 as cv
import matplotlib.pyplot as pltfrom 
from matplotlib import pyplot as plt
from tkinter import *

def open_lenna_color_img():
    
    input_img = cv.imread("Lenna_512_color.tif")
    # color_img = cv.cvtColor(input_img, cv.COLOR_BGR2RGB)
    return input_img
    

def display_original_img():
    color_img = open_lenna_color_img()
    cv.imshow("orginal", color_img)
    # plt.show()
def display_red_and_green_and_blue():
    color_image = open_lenna_color_img()
    color_image = cv.cvtColor(color_image, cv.COLOR_BGR2RGB)

    red_only_image = color_image.copy()
    red_only_image[:, :, 0] = 0
    red_only_image[:, :, 1] = 0

    green_only_image = color_image.copy()
    green_only_image[:, :, 0] = 0
    green_only_image[:, :, 2] = 0

    blue_only_image = color_image.copy()
    blue_only_image[:, :, 1] = 0
    blue_only_image[:, :, 2] = 0

    
    

    plt.subplot(221),plt.imshow(color_image, cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])

    plt.subplot(222),plt.imshow(red_only_image, cmap = 'gray')
    plt.title('Red_only_image'), plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(green_only_image, cmap = 'gray')
    plt.title('Green_only_image'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(224),plt.imshow(blue_only_image, cmap = 'gray')
    plt.title('Blue_only_image'), plt.xticks([]), plt.yticks([])

    plt.show()


def convert_RGB_to_HSI():
    color_img = open_lenna_color_img()
    hsi_img = cv.cvtColor(color_img.copy(), cv.COLOR_RGB2HSV)

    hue_val = hsi_img[:, :, 0]
    saturation_val = hsi_img[:, :, 1]
    intensity_val = hsi_img[:, :, 2]

    color_img = cv.cvtColor(color_img, cv.COLOR_BGR2RGB)
    plt.subplot(231),plt.imshow(color_img, cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])

    plt.subplot(232),plt.imshow(hsi_img, cmap = 'gray')
    plt.title('HSI Model'), plt.xticks([]), plt.yticks([])
    plt.subplot(233),plt.imshow(hue_val, cmap = 'hsv')
    plt.title('Hue Image'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(234),plt.imshow(saturation_val, cmap = 'gray')
    plt.title('Saturation Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(235),plt.imshow(intensity_val, cmap = 'gray')
    plt.title('Intensity Image'), plt.xticks([]), plt.yticks([])

    plt.show()


def color_complement_RGB():
    color_image = open_lenna_color_img()

    complement_img = 255 - color_image 
    cv.imshow("complement", complement_img)


def median_filter(img, filter_kernel):
    temp = []
    
    indexer = filter_kernel // 2
    median_img = np.zeros(img.shape, dtype=np.uint8)

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
            
            temp = sorted(temp)
            median_img[i][j] = temp[len(temp) // 2]
            temp = []
    
    
    return median_img
def smoothing_color_img():
    
    color_image = open_lenna_color_img()
    color_image = cv.cvtColor(color_image, cv.COLOR_BGR2RGB)

    # HSI MODEL get inetnsity only
    hsi_img = cv.cvtColor(color_image.copy(), cv.COLOR_RGB2HSV)
    intensity_img = hsi_img[:, :, 2]
    hsi_img_change = hsi_img.copy()

    hsi_smoothed = cv.blur(intensity_img, (5, 5))
    hsi_img_change[:, :, 2] = hsi_smoothed
    hsi_img_change = cv.cvtColor(hsi_img_change, cv.COLOR_HSV2RGB)

    #RGB MODEL smooth all
    rgb_img = color_image.copy()
    rgb_smoothed = cv.blur(rgb_img, (5, 5))


    
    plt.subplot(221),plt.imshow(color_image, cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])

    # plt.subplot(222),plt.imshow(hsi_img, cmap = 'gray')
    # plt.title('HSI Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(223),plt.imshow(hsi_img_change, cmap = 'gray')
    plt.title('smoothed HSI Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(224),plt.imshow(rgb_smoothed, cmap = 'gray')
    plt.title('smoothed RGB Image'), plt.xticks([]), plt.yticks([])

    plt.show()
def laplacian_filter(input_img):
    kernel = np.asarray([[1,1,1], [1,-8, 1], [1,1,1]])
    des_16S = cv.filter2D(input_img, ddepth=cv.CV_16SC1, kernel=kernel, borderType=cv.BORDER_DEFAULT)

    lap = input_img - des_16S
    lap[lap<0] = 0
    lap[lap>255] = 255

    return lap
    
def sharpening_color_img():
    color_image = open_lenna_color_img()
    color_image = cv.cvtColor(color_image, cv.COLOR_BGR2RGB)

    # HSI MODEL get inetnsity only
    hsi_img = cv.cvtColor(color_image.copy(), cv.COLOR_RGB2HSV)
    intensity_img = hsi_img[:, :, 2]
    hsi_img_change = hsi_img.copy()
    
    hsi_img_change[:, :, 2] = laplacian_filter(intensity_img)
    hsi_img_change = cv.cvtColor(hsi_img_change, cv.COLOR_HSV2RGB)


    # RGB MODEL
    rgb_sharpened = color_image.copy()
    rgb_sharpened = laplacian_filter(rgb_sharpened)


    plt.subplot(221),plt.imshow(color_image, cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])

    # plt.subplot(222),plt.imshow(hsi_img, cmap = 'gray')
    # plt.title('HSI Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(223),plt.imshow(hsi_img_change, cmap = 'gray')
    plt.title('sharpened HSI Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(224),plt.imshow(rgb_sharpened, cmap = 'gray')
    plt.title('sharpened RGB Image'), plt.xticks([]), plt.yticks([])

    plt.show()




def leather_mask():
    color_image = open_lenna_color_img()
    color_image = cv.cvtColor(color_image, cv.COLOR_BGR2RGB)
    hsi_image = cv.cvtColor(color_image, cv.COLOR_RGB2HSV)

   
    # hue 130 - 160
    # saturation > 200

    lower_mask = hsi_image[:,:,0] > 130
    upper_mask = hsi_image[:,:,0] < 160
    saturation = hsi_image[:,:,1] > 20
    
    mask = lower_mask*upper_mask*saturation

    red = color_image[:,:,0]*mask 
    green = color_image[:,:,1]*mask
    blue = color_image[:,:,2]*mask 
    img_masked = np.dstack((red,green,blue))

   
    final = cv.bitwise_and(color_image, img_masked)
    
    
    plt.subplot(221),plt.imshow(color_image, cmap = 'hsv')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
     # show hue and saturation colorbar
    plt.subplot(222),plt.imshow(hsi_image[:, :, 0], cmap = 'hsv')
    plt.title('HUE Distribution'), plt.xticks([]), plt.yticks([])
    

    plt.subplot(223),plt.imshow(hsi_image[:, :, 1], cmap = 'gray')
    plt.title('SATURATION Distribution'), plt.xticks([]), plt.yticks([])
   

    plt.subplot(224),plt.imshow(final, cmap = 'gray')
    plt.title('Final'), plt.xticks([]), plt.yticks([])
    

    plt.show()

    # display_as_hsv(color_image)
    
 


def main():
    window = Tk()
    window.title("DIP COLOR IMAGE")
    window.geometry('300x300')

    # buttons trigger function
    display_origin_btn = Button(window, text="Original Image", command=display_original_img)
    display_origin_btn.grid()

    display_RGB_btn = Button(window, text="RGB Image", command=display_red_and_green_and_blue)
    display_RGB_btn.grid()

    convert_RGB_to_HSI_btn = Button(window, text="HSI Image", command=convert_RGB_to_HSI)
    convert_RGB_to_HSI_btn.grid()

    color_complement_RGB_btn = Button(window, text="Complement", command=color_complement_RGB)
    color_complement_RGB_btn.grid()

    smoothing_color_img_btn = Button(window, text="Smoothing", command=smoothing_color_img)
    smoothing_color_img_btn.grid()

    sharpening_color_img_btn = Button(window, text="Sharpening", command=sharpening_color_img)
    sharpening_color_img_btn.grid()

    leather_mask_btn = Button(window, text="Leather Mask", command=leather_mask)
    leather_mask_btn.grid()

    window.mainloop()






if __name__ == "__main__":
    
    main()