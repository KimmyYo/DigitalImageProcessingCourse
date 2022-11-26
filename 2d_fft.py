import numpy as np
import cv2 as cv

from matplotlib import pyplot as plt



def main():
    lena_img = cv.imread("lenna.tif")
    
    lena_img = cv.cvtColor(lena_img, cv.COLOR_BGR2GRAY)
    # applying 2d-fft to lenna image
    f = np.fft.fft2(lena_img)
    fshift = np.fft.fftshift(f)


    # magnitude spectrum
    mag = np.abs(fshift)
    phase = np.angle(fshift)

    magonly = np.abs(fshift)
    phaseonly = fshift / np.abs(fshift)

    magnitude_spectrum = 10*np.log(magonly).astype(np.uint8)


    magnitude_only = np.abs(np.fft.ifft2(magnitude_spectrum))
    phase_only = np.abs(np.fft.ifft2(phaseonly))
    
    plt.subplot(221),plt.imshow(lena_img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(223),plt.imshow(magnitude_only, cmap = 'gray')
    plt.title('Magnitude Only'), plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(phase_only, cmap = 'gray')
    plt.title('Phase Only'), plt.xticks([]), plt.yticks([])
    
    plt.show()
    


if __name__ == "__main__":
    main()
