import cv2
import matplotlib.pyplot as plt
import numpy as np

##### To-do #####

def fm_spectrum(img):
    """
    Get frequency magnitude spectrum image of input image.
    Returning spectrum image is shifted to center
    Input image should be gray scaled

    :param img: input image
    :return: frequency magnitude spectrum
    """

    # fourier transform 2d using numpy library
    f = np.fft.fft2(img)

    # variables initialization
    n_row, n_col = f.shape
    temp = np.zeros(f.shape, dtype='complex')

    # for every pixel in a frequency domain of input image,
    #   [         |         ]
    #   [    (1)  |   (2)   ]
    #   [-------------------]
    #   [    (3)  |   (4)   ]
    #   [         |         ]
    # swaps the first quadrant (1) with the fourth(4) and the second quadrant (2) with the third(3).
    for row in range(n_row):
        for col in range(n_col):

            # swap first quadrant with the fourth
            if row < n_row // 2 and col < n_col // 2:
                temp[row + n_row // 2][col + n_col // 2] = f[row][col]
            # swap second quadrant with the third
            elif row < n_row // 2 and col > n_col // 2:
                temp[row + n_row // 2][col - n_col // 2] = f[row][col]
            # swap third quadrant with the second
            elif row > n_row // 2 and col < n_col // 2:
                temp[row - n_row // 2][col + n_col // 2] = f[row][col]
            # swap fourth quadrant with the first
            else:
                temp[row - n_row // 2][col - n_col // 2] = f[row][col]

    # get the spectrum and return
    spectrum = np.log(1 + np.abs(temp.real))
    return spectrum

def low_pass_filter(img, th=20):
    """
    Get filtered image that pass through with low-pass filter for an input image.
    :param img: input image
    :param th:  threshold unit in piexel
    :return:  filtered image that pass through with low pass filter
    """
    # Write low pass filter here

    # distance function
    def distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # get magnitude spectrum using fftshift_diy
    spectrum = fftshift_diy(img)

    # variables initialization
    n_row, n_col = img.shape
    center_row, center_col = n_row / 2, n_col / 2

    # create low pass filter
    filter = np.zeros((n_row, n_col))

    # create the low pass filter - for every pixel in the filter,
    for row in range(n_row):
        for col in range(n_col):

            # if the distance between center point and current point is smaller than th parameter,
            # set current point value to 1, which means pass
            if distance(row, col, center_row, center_col) < th:
                filter[row, col] = 1

    # mask the filter on spectrum
    spectrum = spectrum * filter

    # Reform the image from low pass filtered spectrum and return the result image
    f_ishift = ifftshift_diy(spectrum)
    img_filtered= np.fft.ifft2(f_ishift)

    return img_filtered.real

def high_pass_filter(img, th=30):
    """
    Get filtered image that pass through with high-pass filter for an input image.
    :param img: input image
    :param th:
    :return:
    """
    # distance function
    def distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # get magnitude spectrum using fftshift_diy
    spectrum = fftshift_diy(img)

    # variables initialization
    n_row, n_col = img.shape
    center_row, center_col = n_row / 2, n_col / 2

    # create high-pass filter
    filter = np.ones((n_row, n_col))

    # create the high pass filter - for every pixel in the filter,
    for row in range(n_row):
        for col in range(n_col):

            # if the distance between center point and current point is smaller than th parameter,
            # set current point value to  0, which means reject
            if distance(row, col, center_row, center_col) < th:
                filter[row, col] = 0

    # mask the filter on spectrum
    spectrum = spectrum * filter

    f_ishift = ifftshift_diy(spectrum)
    img_filtered = np.fft.ifft2(f_ishift)

    # Reform the image from high pass filtered spectrum and return the result image
    return (img_filtered.real)

def denoise1(img):
    """
    Denoise checker effect with given sample image. (task2_corrupted_1.png)
    Only works for given sample image
    :param img: sample image
    :return: de-noised sample image
    """
    # get the spectrum of sample image
    spectrum = fftshift_diy(img)

    # variable initialization
    n_row, n_col = img.shape


    # Denoise checker image by changing values of points to 0, which are estimated as checker noises
    for row in range(100,n_row, 110):
        for col in range(100,n_col,110):
            for i in range(15):
                for p in range(15):
                    spectrum[row-i][col-p] = 0
    for row in range(155, n_row- 100, 220):
        for col in range(155, n_col- 100, 220):
            for i in range(15):
                for p in range(15):
                    spectrum[row - i][col - p] = 0

    # reform image from de-noised spectrum and return the de-noised image
    f_ishift = ifftshift_diy(spectrum)
    img_filtered = np.fft.ifft2(f_ishift)

    return img_filtered.real

def denoise2(img):
    """
    De-noise wave effect with given sample image (task2_corrupted_2.png)
    Only works for given sample image
    :param img: sample image
    :return: de-noised sample image
    """

    # distance function
    def distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # get the spectrum from sample image
    spectrum = fftshift_diy(img)

    # variables initialization
    n_row, n_col = img.shape
    center_row, center_col = n_row / 2, n_col / 2

    # create band-reject filter
    filter = np.zeros((n_row, n_col))

    for row in range(n_row):
        for col in range(n_col):
            if distance(row, col, center_row, center_col) < 40:
                filter[row, col] = 1
            elif 43 < distance(row, col, center_row, center_col):
                filter[row, col] = 1

    # mask the band-reject filter to spectrum
    spectrum = spectrum * filter

    # reform image from de-noised spectrum and return de-noised image
    f_ishift = ifftshift_diy(spectrum)
    img_filtered = np.fft.ifft2(f_ishift)

    return img_filtered.real

def fftshift_diy(img):
    """
    DIY version of numpy.fft.fftshift
    Created for this project
    This project prohibits using numpy.fft.fftshift library function
    Further explanation is written in comment lines of fm_specturm function

    :param img: input image
    :return: same as numpy.fft.fftshift(img)
    """

    # do the fourier 2d transform using numpy library
    f = np.fft.fft2(img)

    # initialization
    n_row, n_col = f.shape
    temp = np.zeros(f.shape, dtype='complex')

    # shift!
    for row in range(n_row):
        for col in range(n_col):
            if row < n_row // 2 and col < n_col // 2:
                temp[row + n_row // 2][col + n_col // 2] = f[row][col]
            elif row < n_row // 2 and col > n_col // 2:
                temp[row + n_row // 2][col - n_col // 2] = f[row][col]
            elif row > n_row // 2 and col < n_col // 2:
                temp[row - n_row // 2][col + n_col // 2] = f[row][col]
            else:
                temp[row - n_row // 2][col - n_col // 2] = f[row][col]

    return temp

def ifftshift_diy(spectrum):
    """
       DIY version of numpy.fft.ifftshift
       Created for this project
       This project prohibits using numpy.fft.ifftshift library function
       Further explanation is written in comment lines of fm_specturm function

       :param img: input image
       :return: same as numpy.fft.ifftshift(img)
       """
    # initialization
    n_row, n_col = spectrum.shape
    temp = np.zeros(spectrum.shape, dtype='complex_')

    # shift 
    for row in range(n_row):
        for col in range(n_col):
            if row < n_row // 2 and col < n_col // 2:
                temp[row + n_row // 2][col + n_col // 2] = spectrum[row][col]
            elif row < n_row // 2 and col >= n_col // 2:
                temp[row + n_row // 2][col - n_col // 2] = spectrum[row][col]
            elif row >= n_row // 2 and col < n_col // 2:
                temp[row - n_row // 2][col + n_col // 2] = spectrum[row][col]
            else:
                temp[row - n_row // 2][col - n_col // 2] = spectrum[row][col]

    return temp
########################################################################################################################

if __name__ == '__main__':
    img = cv2.imread('task2_sample.png', cv2.IMREAD_GRAYSCALE)
    cor1 = cv2.imread('task2_corrupted_1.png', cv2.IMREAD_GRAYSCALE)
    cor2 = cv2.imread('task2_corrupted_2.png', cv2.IMREAD_GRAYSCALE)

    def drawFigure(loc, img, label):
        plt.subplot(*loc), plt.imshow(img, cmap='gray')
        plt.title(label), plt.xticks([]), plt.yticks([])

    drawFigure((2,7,1), img, 'Original')
    drawFigure((2,7,2), (low_pass_filter(img)), 'Low-pass')
    drawFigure((2,7,3), (high_pass_filter(img)), 'High-pass')
    drawFigure((2,7,4), cor1, 'Noised')
    drawFigure((2,7,5), denoise1(cor1), 'Denoised')
    drawFigure((2,7,6), cor2, 'Noised')
    drawFigure((2,7,7), denoise2(cor2), 'Denoised')

    drawFigure((2,7,8), fm_spectrum(img), 'Spectrum')
    drawFigure((2,7,9), fm_spectrum(low_pass_filter(img)), 'Spectrum')
    drawFigure((2,7,10), fm_spectrum(high_pass_filter(img)), 'Spectrum')
    drawFigure((2,7,11), fm_spectrum(cor1), 'Spectrum')
    drawFigure((2,7,12), fm_spectrum(denoise1(cor1)), 'Spectrum')
    drawFigure((2,7,13), fm_spectrum(cor2), 'Spectrum')
    drawFigure((2,7,14), fm_spectrum(denoise2(cor2)), 'Spectrum')

    plt.show()