# Allen Kim
# CS 519
# Assignment 2, #3


import cv2
import matplotlib.pyplot as plt
import numpy as np
from generate_random_color import generate_new_color as gnc


def segment_intensity(k, img, out_img):
    """ Loads an image, performs kmeans clustering on intensity and outputs an image
        where each pixel is replaced by one with the same value as its cluster center.

    :param k: int: number of clusters to find
    :param img: input image
    :param out_img: output image path
    :return: void
    """
    im = cv2.imread(img, 0)                                         # read image as grayscale
    pixels = np.float32(im.ravel())                                 # convert to 1D array of pixels
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)    # define parameters for kmeans
    flags = cv2.KMEANS_RANDOM_CENTERS                                           # define parameters for kmeans
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 50, flags)       # run cv2.kmeans on the image
    centers = np.uint8(centers)                                    # convert centers to uint8
    reassigned_pixels = centers[labels.flatten()]       # make a new array out of values from centers. labels tells
                                                        # us which center has been assigned to each pixel.
    result_image = reassigned_pixels.reshape(im.shape)  # reshape to get result_image

    plt.figure()                                                # plt to display result
    plt.xticks([])
    plt.yticks([])
    plt.imshow(result_image, cmap='gray', vmin=0, vmax=255)

    cv2.imwrite(out_img, result_image)                          # output to file


def segment_int_loc(k, img, out_img):
    """ Performs segmentation based on intensity and location. Each pixel is represented as a feature
        vector of [row, column, intensity]. Displays the segmented image with a different color for each
        segment.

    :param k: int: number of clusters to find
    :param img: input image
    :param out_img: output image path
    :return: void
    """
    im = cv2.imread(img, 0)  # read image as grayscale

    pixels = np.zeros((im.shape[0] * im.shape[1], 3), np.float32)
    index = 0
    for (r, c), intensity in np.ndenumerate(im):
        pixels[index] = np.asarray([intensity, r, c])
        index += 1

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # define parameters for kmeans
    flags = cv2.KMEANS_RANDOM_CENTERS  # define parameters for kmeans
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 50, flags)  # run cv2.kmeans on the image
    centers = np.uint8(centers)  # convert centers to uint8

    colors = []
    for i in range(k):
        colors.append(gnc(colors, pastel_factor=0.6))
    normed_colors = [np.uint8(np.asarray(color) * 255) for color in colors]

    reassigned_pixels = np.asarray([normed_colors[label] for label in labels.flatten()])

    # reassigned_pixels = normed_colors[labels.flatten()]  # make a new array out of values from centers. labels tells
    # us which center has been assigned to each pixel.
    result_image = reassigned_pixels.reshape((im.shape[0], im.shape[1], 3))  # reshape to get result_image

    plt.figure()  # plt to display result
    plt.xticks([])
    plt.yticks([])
    plt.imshow(result_image)

    cv2.imwrite(out_img, result_image)                          # output to file


if __name__ == '__main__':
    segment_intensity(5, '../images/3/test5.jpg', '../images/3/test5_intens_k5.png')
    segment_int_loc(8, '../images/3/test5.jpg', '../images/3/test5_int_loc_k8.png')

    segment_intensity(5, '../images/3/test6.png', '../images/3/test6_intens_k5.png')
    segment_int_loc(10, '../images/3/test6.png', '../images/3/test6_int_loc_k10.png')

    segment_intensity(16, '../images/3/test2.png', '../images/3/test2_intens_k16.png')
    segment_int_loc(16, '../images/3/test2.png', '../images/3/test2_int_loc_k16.png')

    plt.show()