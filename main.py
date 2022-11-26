import numpy as np
import cv2 as cv
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import math
import matplotlib.pyplot as plt


F_TYPES = [('Jpg Files', '*.jpg'), ('TIF Files','*.tif'),  ('TIFF Files','*.tiff')]


# config opencv image to image on tkinter canvas
def config_canvas_image(cv_image_to_change):

    pil_image = ImageTk.PhotoImage(image=Image.fromarray(cv_image_to_change))

    cv.imwrite(('images/your_image.jpg'), cv_image_to_change)
    canvas.itemconfig(image_sprite, image=pil_image)
    canvas['image'] = pil_image



# Open/Save/Display image
def open_image():
    
    img_path = askopenfilename(filetypes=F_TYPES)
    global cv_image
    cv_image = cv.imread(img_path)
   
    height, width = cv_image.shape[:2]
    if (width > 580) or (height > 1000):
        cv_image = cv.resize(cv_image, (width//2, height//2), interpolation=cv.INTER_CUBIC)
        

    # turn into PIL image
    global pil_image
    pil_image = ImageTk.PhotoImage(image=Image.fromarray(cv_image))
    
    open_image_label.config(text='Updated!')
    canvas.itemconfig(image_sprite, image=pil_image)
    canvas['image'] = pil_image

   

    

def save_image():
    # filename = asksaveasfile(mode='w', filetypes=F_TYPES)
    # save in local folder
    cv.imwrite(('images/your_image.jpg'), cv_image)
    open_image_label.config(text='Saved!')
    


def rotate_image():
    
    degree = int(rotate_entry.get())
    height, width = cv_image.shape[:2]
    rotated = cv_image
    width += 10
    R = cv.getRotationMatrix2D((height//2, width//2), angle=degree, scale=1)
    rotated = cv.warpAffine(cv_image, R, (height, width))
    cv.imwrite("images/rotated_img.jpg", rotated)
    config_canvas_image(rotated)



# Adjust Contrast/Brightness of image by changing values of a and b 
# get contrast and brightness value from tkinter scale
def get_contrast_brightness():
    contrast_level = int(contrast_scale.get())
    brightness_level = int(brightness_scale. get()) 

    return contrast_level, brightness_level

# Linearly
def linearly(): # contrast -> a, brightness -> b

    contrast_level, brightness_level = get_contrast_brightness()
    
    linear_change = cv_image
    linear_change = np.clip(cv_image * (contrast_level/127 + 1) - contrast_level + brightness_level, 0, 255)
    linear_change = np.uint8(linear_change)
   
    cv.imwrite("images/inear_img.jpg", linear_change)
    config_canvas_image(linear_change)

# Exponentially
def exponentially(): # contrast -> a, brightness -> b

    contrast_level, brightness_level = get_contrast_brightness()
    contrast_level /= 20

    if contrast_level < 0:
        contrast_level = 1/(-contrast_level)
    elif contrast_level == 0: 
        contrast_level = 1
    
    brightness_level /= 102
    
    gamma = np.clip(contrast_level - brightness_level, 0, 6)
    
    exp_change = np.zeros(cv_image.shape, cv_image.dtype)
    # exp_change = cv.convertScaleAbs(cv_image, alpha=contrast_level, beta=brightness_level)
  
    exp_change = np.array(255*(cv_image/255)**(contrast_level - brightness_level), dtype='uint8')
    
    cv.imwrite("images/exp_img.jpg", exp_change)
    config_canvas_image(exp_change)

# Logarithmically
def log_transform():

    contrast_level, brightness_level = get_contrast_brightness()

    contrast_level = (contrast_level + 101) / 20
    brightness_level  = (brightness_level + 101) / 100
    

    
    slope = contrast_level

    log_change = np.zeros(cv_image.shape, dtype='uint8')
    c = 255 / math.log(1 + np.max(cv_image), slope)
    log_change = np.array(brightness_level * c * np.log((cv_image + 1)/np.log(slope)), dtype=np.uint8)
    # log_change = np.array((np.log(cv_image+1)/(np.log(1+np.max(cv_image))))*255, dtype=np.uint8)
   
    cv.imwrite("images/log_img.jpg", log_change)
    config_canvas_image(log_change)
     
    
# Zoom in and Shrink -> bilinear interpolation
def zoom_in_bi():
    x = y = zoom_in_entry.get()
    
    cropped = cv_image[0:600, 0:600]
    bilinear_image = cv.resize(cropped, None,  fx = x, fy = y, interpolation = cv.INTER_LINEAR)
    
    cv.imwrite('images/zoomin_img.jpg', bilinear_image)
    config_canvas_image(bilinear_image)

def shrink():
    val = zoom_in_entry.get()

    shrink_img = cv.resize(cv_image, (cv_image.shape[1]//val, cv_image.shape[0]//val), interpolation=cv.INTER_AREA)

    cv.imwrite('images/shrink_img.jpg', shrink_img)
    config_canvas_image(shrink_img)


# Gray level slicing 
def gray_level_slicing():
    min_range = int(gray_min_entry.get())
    max_range = int(gray_max_entry.get())

    remain_color = True
    if blk.get():
        remain_color = False
    
    row, column = cv_image.shape[:2]

    gray_slice_img = np.zeros((row, column), dtype='uint8')

    for i in range(row):
        for j in range(column):

            if cv_image[i, j][0] > min_range and cv_image[i, j][0] < max_range:
                gray_slice_img[i, j] = 255
            
            else: 
                if not remain_color:
                    gray_slice_img[i, j] = 0
                else:
                    gray_slice_img[i, j] = cv_image[i, j][0]
    
    cv.imwrite("images/gray_slicing_img.jpg", gray_slice_img)
    config_canvas_image(gray_slice_img)



# display histogram chart by plt show
def display_histogram(image, x, y1):
   
    new_frequency = {i:0 for i in range(256)}
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_val = image[i, j][0]
            new_frequency[pixel_val] += 1

    y2 = list(new_frequency.values())
    plt.bar(x, y1, color="blue", label="original image")
    plt.bar(x, y2, color="orange", label="equalized image")
    plt.xlabel("Gray Scale Level")
    plt.ylabel("Frequency")
    plt.title("Image Histogram")
    plt.show()
    
# histogram equlization function
def histogram_equalization():
   
    # calculate frequency
    origin_frequency = {i: 0 for i in range(256)}
    height, width = cv_image.shape[:2]
    
    for i in range(height):
        for j in range(width):
            pixel_val = cv_image[i, j][0]
            origin_frequency[pixel_val] += 1

    
    probability = [i / (width*height) for i in list(origin_frequency.values())]

    # cumulative distribution
    total = 0
    cdf = []
    for cum in probability:
        total += cum
        cdf.append(round(total*255))
    print(cdf)

    # remap values 
    for i in range(width):
        for j in range(height):
            g = cv_image[i, j][0]
            cv_image[i, j] = cdf[g]

      
    cv.imwrite('images/histogram_equalized_img.jpg', cv_image)
    display_histogram(cv_image, origin_frequency.keys(), origin_frequency.values())
    config_canvas_image(cv_image)


# bit plan function
def bit_plane():   

    height, width = cv_image.shape[:2]

    # create a list of original pixel values
    binary_pixel_values = []
    for i in range(width):
        for  j in range(height):
            # turn the values into binary form
            binary_pixel_values.append(np.binary_repr(cv_image[i, j][0], width=8))
    
    # get user nth bit value
    nth_bit = int(nth_bit_input.get())
    
    factor = 2 ** (nth_bit - 1)
    bit_plane_image = (np.array([int(binary_value[-nth_bit + 8])for binary_value in binary_pixel_values], dtype=np.uint8)*factor).reshape(width, height)

    cv.imwrite('images/bit_plane_img.jpg', bit_plane_image)
    config_canvas_image(bit_plane_image)
   

# def mean_filter(img):
#     mean_img = cv.GaussianBlur(img, (3,3), 1)
    
#     return mean_img

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

# smoothin image function
def smoothing_image():

    # user defined degree would be the size of the kernel (x*y)
    degree = int(smooth_degree.get())
    smoothing_filtered_img = cv.blur(cv_image, (degree, degree))
    cv.imwrite('images/smoothing_img.jpg', smoothing_filtered_img)
    config_canvas_image(smoothing_filtered_img)
        
      

# unsharp mask 
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):

    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)

    return sharpened


# sharpening image function
def sharpening_image():
    degree = int(sharp_degree.get())
    sharpened_img = unsharp_mask(cv_image, amount=degree)

    cv.imwrite('images/sharpened_img.jpg', sharpened_img)
    config_canvas_image(sharpened_img)


# applying laplacian filter from best image choosen from compare_four_imgs -> pirate_a applyed median filter
# read best raw img first 
def read_raw_img(img):

    fd = open(img, 'rb')
    rows = 512
    cols = 512
    f = np.fromfile(fd, dtype = np.uint8, count = rows * cols)
    raw_img = f.reshape((rows, cols)) # notice row, column format
    fd.close()
    

    return raw_img

def laplacian_filter():
    
    best_img = read_raw_img('pirate_a.raw')
    best_blurred = median_filter(best_img, 3)
    
    lap = cv.Laplacian(best_blurred, cv.CV_16S, ksize=3).astype(np.uint8)
    lap = cv.convertScaleAbs(lap)

    cv.imwrite('images/laplacian_img.jpg', lap)
    config_canvas_image(lap)



def mean_filter():
    height, width = cv_image.shape[:2]
    w = 1
    for i in range(w, width-w):
        for j in range(w, height-w):
            block = cv_image[i-w:i+w+1, j-w:j+w+1]
            average = np.mean(block, dtype=np.float32)
            cv_image[i, j] = int(average)

    cv.imwrite('images/mean_img.jpg', cv_image)
    config_canvas_image(cv_image)

def fft_2d(img):
    
    N,M = np.shape(img)
    i, j = np.meshgrid(np.arange(M), np.arange(N))
    mult_factor = np.power( np.ones((N,M)) * -1 , i + j )
    tmp = img * mult_factor
    # print("Calculating DFT")
    dft = np.fft.fft2(tmp)
    # print("Calculating inverse DFT")
    conj_img = dft.conj()
    idft = np.fft.ifft2(conj_img)
    out_img = np.abs((mult_factor * idft.real) + (1j * idft.imag))
    fft_image = np.fft.fftshift(np.fft.fft2(out_img))
    conj = np.conj(fft_image)
    ifft_image = np.fft.fftshift(np.fft.ifft(conj))
    
    final_image = np.abs((10*ifft_image.real) + 1j * ifft_image.imag)
    final_image = np.array(idft, dtype=np.uint8)
    # magnitude_spectrum = 10*np.log(np.abs(ifft_image) + 1)
    # magnitude_spectrum = np.array(magnitude_spectrum, dtype=np.uint8)


    return final_image
    # config_canvas_image(magnitude_spectrum)
    # phase_spectrum = np.angle(fshift)
    # phase_spectrum = np.array(phase_spectrum, dtype=np.uint8)
    # config_canvas_image(phase_spectrum)
def inverse_fft_2d(input_image):
    inversed = np.fft.ifft2(input_image)
    

def minus_1_xy(input_image):

    # multiply the image by (-1)^x+y
    shifted = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
    shifted = np.fft.fftshift(shifted)

    return shifted

def dft(input_image):

    height, width = cv_image.shape[:2]
   
    t = np.zeros((width, height), complex)
    dft_image = np.zeros((width, height), complex)
    # computing the DFT - applying 1d-fft to reduce the executing time
    m = np.arange(height)
    n = np.arange(width)
    x = m.reshape((height, 1))
    y = n.reshape((width, 1))
    
    for row in range(width):
        M1 = 1j * np.sin(-2*np.pi*y*n/height) + np.cos(-2*np.pi*y*n/height)
        t[row] = np.dot(M1, input_image[row])

    for col in range(height):
        M2 = 1j * np.sin(-2*np.pi*x*m/height) + np.cos(-2*np.pi*x*m/height)
        dft_image[:, col] = np.dot(M2, t[:, col])

    dft_image = np.log(np.abs(dft_image)+1).astype(np.uint8)
    # dft_image = np.fft.fftshift(dft_image).astype(np.uint8)
    dft_image = cv.cvtColor(dft_image, cv.COLOR_GRAY2BGR)
    # print(dft_image)
    config_canvas_image(dft_image)
    

    

def frequency_domain_filtering():
    shifted = minus_1_xy(cv_image)
    dft_image = fft_2d(shifted)
    # conj = np.conjugate(dft_image)
    # dft_image = np.fft.ifft2(conj)
    final = minus_1_xy(cv.cvtColor(dft_image, cv.COLOR_GRAY2BGR))
    config_canvas_image(final)


# GUI
window = Tk()
window.title("DIP")
window.geometry('1920x1080')



initial_img = Image.open("images/add-image.png")
initial_img.rotate(180, expand=True)
image_size = (initial_img.width, initial_img.height)
resized_PILimage= initial_img.resize((initial_img.width//10, initial_img.height//10))
image = ImageTk.PhotoImage(resized_PILimage)

spc_label = Label(window, text='', padx=200)
spc_label.grid(row=1, column=5, columnspan=20)

# Canvas setup
canvas = Canvas(window, width=580, height=580, highlightthickness=0, borderwidth=3, bg="lightgray")
canvas.grid(row=5, column=30, rowspan=20, columnspan=100)
image_sprite = canvas.create_image(290, 290, image=image)


# FILE 
file_label = Label(window, text='File', font=(("Arial", 24, "underline")), fg='black', padx=20)
file_label.grid(row=0, column=0)

open_image_label = Label(window, text='', font=(("Times New Roman", 15, "bold")), fg="black")
open_image_label.grid(row=3, column=0)

open_image_button = Button(window, text='open', command=open_image)
open_image_button.grid(row=1, column=0)

save_image_button = Button(window, text='Save', command=save_image)
save_image_button.grid(row=2, column=0)



# ROTATE
rotate_label = Label(window, text='Rotate', font=(("Arial", 24, "underline")), fg='black', padx=20)
rotate_label.grid(row=0, column=1)

rotate_entry = Entry(window, width=7)
rotate_entry.grid(row=1, column=1)

rotate_button = Button(window, text='Rotate', command=rotate_image)
rotate_button.grid(row=2, column=1)



# ENHANCE
enhance_label = Label(window, text='Enhance', font=(("Arial", 24, "underline")), fg='black')
enhance_label.grid(row=5, column=0, columnspan=2)

contrast_label = Label(window, text='contrast', font=(("Times New Roman", 18, "bold")))
contrast_label.grid(row=6, column=0, sticky=S)

contrast_scale = Scale(window, orient=HORIZONTAL, from_=-100, to=100, resolution=1)  # scale can't be negative
contrast_scale.grid(row=6, column=1)

brightness_label = Label(window, text='brightness', font=(("Times New Roman", 18, "bold")))
brightness_label.grid(row=7, column=0, sticky=S)

brightness_scale = Scale(window, orient=HORIZONTAL, from_=-100, to=100, resolution=1)
brightness_scale.grid(row=7, column=1)


linear_button = Button(window, text="Linearly", command=linearly)
linear_button.grid(row=9, column=0, sticky='w')

exponent_button = Button(window, text="Exponentially", command=exponentially)
exponent_button.grid(row=9, column=1, sticky='w')

log_button = Button(window, text="Logarithm", command=log_transform)
log_button.grid(row=9, column=2, sticky='w')




# ZOOM IN AND SHRINK
zm_label = Label(window, text='Scaling', font=("Arial", 24, 'underline'), pady=10)
zm_label.grid(row=13, column=0, columnspan=2)

scale_label = Label(window, text='scale', font=("Times New Roman", 18, 'bold'))
scale_label.grid(row=14, column=0, sticky=S+E)

zoom_in_entry = Scale(window, orient=HORIZONTAL, from_=1, to=10)
zoom_in_entry.grid(row=14, column=1, sticky=W)

zoomin_button = Button(window, text="Zoom in", command=zoom_in_bi)
zoomin_button.grid(row=14, column=2, sticky=S+W)

shrink_button = Button(window, text="Shrink", command=shrink)
shrink_button.grid(row=14, column=3, sticky=S+W)



# GRAY LEVEL SLICING
gray_slcing_label = Label(window, text='Gray Slicing', font=("Arial", 24, 'underline'), pady=10)
gray_slcing_label.grid(row=16, column=0, columnspan=2)

min_label = Label(window, text='min', font=("Times New Roman", 18, 'bold'))
min_label.grid(row=17, column=0, sticky=E)

gray_min_entry = Entry(window, width=10)
gray_min_entry.grid(row=17, column=1, sticky=W)

max_label = Label(window, text='max', font=("Times New Roman", 18, 'bold'))
max_label.grid(row=18, column=0, sticky=E)

gray_max_entry = Entry(window, width=10)
gray_max_entry.grid(row=18, column=1, sticky=W)

blk = IntVar()
gray_slicing_black = Checkbutton(window, text='black color', variable=blk)
gray_slicing_black.grid(row=19, column=1, sticky=W)

gray_slicing_btn = Button(window, text='Gray Slice', command=gray_level_slicing)
gray_slicing_btn.grid(row=19, column=2, sticky=W)


hist_qual_lbl = Label(window, text="Histogram",font=("Arial", 24, 'underline'))
hist_qual_lbl.grid(row=20, column=0)
display_hist = Button(window, text='display hist', command=histogram_equalization)
display_hist.grid(row=21, column=0)


bit_plane_lbl = Label(window, text='Bit-Plane', font=("Arial", 24, 'underline'))
bit_plane_lbl.grid(row=20, column=1)
nth_bit_input = Entry(window, width=3)
nth_bit_input.grid(row=21, column=1)
bit_plane_button = Button(window, text="Bit Plane", command=bit_plane)
bit_plane_button.grid(row=21, column=2)

s2_label = Label(window, text="Smoothin & Sharpening", font=("Arial", 24, 'underline'))
s2_label.grid(row=22, column=0, columnspan=2)
smooth_degree = Scale(window, width=10, orient=HORIZONTAL, from_=1, to=10)
smooth_degree.grid(row=23, column=0)
smooth_img_btn = Button(window, text="Smoothing", command=smoothing_image)
smooth_img_btn.grid(row=24, column=0)

sharp_degree = Scale(window, width=10, orient=HORIZONTAL, from_=1, to=10)
sharp_degree.grid(row=23, column=1)
sharp_img_btn = Button(window, text="Sharpening", command=sharpening_image)
sharp_img_btn.grid(row=24, column=1)


lap_label = Label(window, text='Laplacian', font=("Arial", 24, 'underline'))
lap_label.grid(row=25, column=0)
laplacian_btn = Button(window, text="Laplacian", command=laplacian_filter)
laplacian_btn.grid(row=26, column=0)


mean_filter = Button(window, text="Mean Filter", command=mean_filter)
mean_filter.grid()

fft_2d_button = Button(window, text="2D-FFT", command=frequency_domain_filtering)
fft_2d_button.grid()

window.mainloop()




