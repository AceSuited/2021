import cv2
import numpy as np
from task1.utils import *
import math


def task1(src_img_path, clean_img_path, dst_img_path):
    """
    This is main function for task 1.
    It takes 3 arguments,
    'src_img_path' is path for source image.
    'clean_img_path' is path for clean image.
    'dst_img_path' is path for output image, where your result image should be saved.

    You should load image in 'src_img_path', and then perform task 1 of your assignment 1,
    and then save your result image to 'dst_img_path'.
    """

    noisy_img = cv2.imread(src_img_path)
    clean_img = cv2.imread(clean_img_path)
    result_img = None
    rms_min = 100
    # do noise removal

    filtered1 = apply_average_filter(noisy_img, 3)
    if calculate_rms_cropped(filtered1, clean_img) < rms_min:
        rms_min = calculate_rms_cropped(filtered1,clean_img)
        result_img = filtered1

    filtered2 = apply_median_filter(noisy_img,3)
    if calculate_rms_cropped(filtered2, clean_img) < rms_min:
        rms_min = calculate_rms_cropped(filtered2,clean_img)
        result_img = filtered2

    filtered3 = apply_bilateral_filter(noisy_img,3,100,30)
    if calculate_rms_cropped(filtered3, clean_img) < rms_min:
        rms_min = calculate_rms_cropped(filtered3,clean_img)
        result_img = filtered3

    filtered4 = apply_bilateral_filter(noisy_img,5,100,30)
    if calculate_rms_cropped(filtered4, clean_img) < rms_min:
        rms_min = calculate_rms_cropped(filtered4,clean_img)
        result_img = filtered4


    cv2.imwrite(dst_img_path, result_img)


def apply_average_filter(img, kernel_size):
    """
    You should implement average filter convolution algorithm in this function.
    It takes 2 arguments,
    'img' is source image, and you should perform convolution with average filter.
    'kernel_size' is a int value, which determines kernel size of average filter.

    You should return result image.
    """

    # variables initialization
    n_row, n_col, n_chan = img.shape
    padding_size = (kernel_size // 2)

    # create a kernel for average filter
    kernel = np.ones((kernel_size, kernel_size))
    kernel = kernel / (kernel_size * kernel_size)

    # create container for result image
    img_new = np.zeros([n_row, n_col, n_chan])

    # for every RGB channel in image
    for chan in range(n_chan):

        # pad zeros to current channel
        current_chan = img[:, :, chan]
        padded_channel = np.zeros((n_row + 2 * padding_size, n_col + 2 * padding_size))
        padded_channel[padding_size:padding_size + n_row, padding_size:padding_size + n_col] = current_chan

        # temporary container variable for result image
        filtered_padded_channel = padded_channel.copy()

        # for every pixels in image,
        for row in range(padding_size, padding_size + n_row):
            for col in range(padding_size, padding_size + n_col):

                # average convolution computation for given pixel
                convolution_area = padded_channel[row - padding_size:row + padding_size + 1, col - padding_size:col + padding_size + 1]
                filtered_value = int((convolution_area * kernel).sum())
                filtered_padded_channel[row, col] = filtered_value

        # save the result of filtered channel to img_new
        img_new[:, :, chan] = filtered_padded_channel[padding_size:padding_size + n_row, padding_size:padding_size + n_col]

    # type conversion and return
    img_new = img_new.astype('uint8')

    return img_new


def apply_median_filter(img, kernel_size):
    """
    You should implement median filter convolution algorithm in this function.
    It takes 2 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is a int value, which determines kernel size of median filter.

    You should return result image.
    """

    # variables initialization
    n_row, n_col, n_chan = img.shape
    padding_size = (kernel_size // 2)

    # create container for result image
    img_new = np.zeros([n_row, n_col, n_chan])

    # for every RGB channel in image
    for chan in range(n_chan):

        # pad zeros to current channel
        current_chan = img[:, :, chan]
        padded_channel = np.zeros((n_row + 2 * padding_size, n_col + 2 * padding_size))
        padded_channel[padding_size:padding_size + n_row, padding_size:padding_size + n_col] = current_chan

        # temporary container variable for result image
        filtered_padded_channel = padded_channel.copy()

        # for every pixels in image,
        for row in range(padding_size, padding_size + n_row):
            for col in range(padding_size, padding_size + n_col):

                # median convolution computation for given pixel
                convolution_area = padded_channel[row - padding_size:row + padding_size + 1, col - padding_size:col + padding_size + 1]
                filtered_value = np.median(convolution_area)
                filtered_padded_channel[row, col] = filtered_value

        # save the result of filtered channel to img_new
        img_new[:, :, chan] = filtered_padded_channel[padding_size:padding_size + n_row, padding_size:padding_size + n_col]

    # type conversion and return
    img_new.astype('uint8')

    return img_new


def apply_bilateral_filter(img, kernel_size, sigma_s, sigma_r):
    """
    You should implement convolution with additional filter.
    You can use any filters for this function, except average, median filter.
    It takes at least 2 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is a int value, which determines kernel size of average filter.
    'sigma_s' is a int value, which is a sigma value for G_s
    'sigma_r' is a int value, which is a sigma value for G_r

    You can add more arguments for this function if you need.

    You should return result image.
    """

    # Gaussian Function
    def gaussian(x, sigma):
        return (1.0 / (2 * np.pi * (sigma ** 2))) * np.exp(- (x ** 2) / (2 * sigma ** 2))

    # Distance Function
    def distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # variables initialization
    n_row, n_col, n_chan = img.shape
    img_new = np.zeros([n_row, n_col, n_chan])

    # for every RGB channel in image
    for chan in range(n_chan):

        # variables initialization
        current_chan = img[:, :, chan]
        filtered_chan = np.zeros([n_row, n_col])

        # for every pixel in an image
        for col in range(n_col):
            for row in range(n_row):
                wp = 0                     # normalization wp
                filtered_value_sum = 0     # sum of Gs x Gr x Iq in convolution area

                # for every pixels in convolution kernel
                for col_kernel in range(-(kernel_size // 2), (kernel_size // 2) + 1):
                    for row_kernel in range(-(kernel_size // 2), (kernel_size // 2) + 1):

                        # calculate current pixel in convolution area
                        col_fPoint = col + col_kernel
                        row_fPoint = row + row_kernel

                        # Only when current pixel coordinates is not out of index, do the convolution calculation
                        if (col_fPoint >= 0) and (col_fPoint < n_col) and (row_fPoint >= 0) and (row_fPoint < n_row):

                            # compute gaussian spatial/range values in this point
                            gaussian_spatial = gaussian(distance(row_fPoint, col_fPoint, row, col), sigma_s)
                            gaussian_range = gaussian(abs(int(current_chan[row_fPoint][col_fPoint]) - int(current_chan[row][col])), sigma_r)

                            w = gaussian_spatial * gaussian_range                               # w = Gs x Gr
                            filtered_value_sum += current_chan[row_fPoint][col_fPoint] * w      # w * Iq
                            wp += w                                                             # wp (variable for normalization)
                filtered_value_norm = filtered_value_sum / wp                                   # normalization

                # save result of computation for corresponding pixel
                filtered_chan[row][col] = int(np.round(filtered_value_norm))

        # save computation result of channel
        img_new[:, :, chan] = filtered_chan


    # type conversion and return
    img_new = img_new.astype('uint8')

    return img_new

