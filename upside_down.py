import numpy as np
import cv2 as cv
import matplotlib.pyplot as pltfrom 
from matplotlib import pyplot as plt



def main():
    
    dip_img = cv.imread("DIP_image.tif")


    # shift first (-1)x+y
    shifted = cv.cvtColor(dip_img, cv.COLOR_BGR2GRAY)
    shifted = np.fft.fftshift(shifted)

    N,M = np.shape(shifted)
    i, j = np.meshgrid(np.arange(M), np.arange(N))
    mult_factor = np.power( np.ones((N,M)) * -1 , i + j )
    tmp = shifted * mult_factor
    # print("Calculating DFT")
    dft = np.fft.fft2(tmp)
    dft_show = 10*np.log(np.abs(np.fft.fftshift(dft)))
    
    # print("Calculating inverse DFT")
    conj_img = dft.conj()
    conj_img_show = np.abs(conj_img)


    idft = np.abs(np.fft.ifft2(conj_img))
    inverse_img = np.abs((mult_factor * idft.real) + (1j * idft.imag))
    
    
    shifted_2 = np.fft.fftshift(inverse_img)
   
    
    plt.subplot(231),plt.imshow(dip_img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(232),plt.imshow(shifted, cmap = 'gray')
    plt.title('Shifted-1'), plt.xticks([]), plt.yticks([])

    plt.subplot(233),plt.imshow(dft_show, cmap = 'gray')
    plt.title('Compute DFT'), plt.xticks([]), plt.yticks([])

    plt.subplot(234),plt.imshow(conj_img_show, cmap = 'gray')
    plt.title('Conjugate'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(235),plt.imshow(inverse_img, cmap = 'gray')
    plt.title('Inverse DFT'), plt.xticks([]), plt.yticks([])

    plt.subplot(236),plt.imshow(shifted_2, cmap = 'gray')
    plt.title('Phase Only'), plt.xticks([]), plt.yticks([])
    
    plt.show()

if __name__ == '__main__':
    main()