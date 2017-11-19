# Allen Kim
# CS 519
# Assignment 2, #2


import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import log2


def make_gp(im):
    """ Creates Gaussian pyramid based on image src

    :param im: source image
    :return: [images] Gaussian pyramid.
             List is ordered from largest to smallest images.
    """
    num_levels = int(np.round(log2(min(im.shape)), 0))     # how many levels is the pyramid?

    src_ = im.copy()                    # make a copy of source image
    pyramid = [src_]                    # 1st element of pyramid is image itself
    for _ in range(num_levels):         # for rest of pyramid:
        src_ = cv2.pyrDown(src_)        # Gaussian blur and downsample
        pyramid.append(src_)            # append to pyramid
    return pyramid                      # return pyramid


def make_lp(gp):
    """ Creates Laplacian pyramid based on a Gaussian pyramid.

    :param gp: [images] a Gaussian pyramid
    :return: [images] the corresponding Laplacian pyramid.
    """
    lp = []                                             # initialize list that we will return
    for i, layer in enumerate(gp[:-1]):                 # use enumerate to get index and layer from Gaussian pyramid
        layer = layer.astype(float)                     # convert to float to allow for negative numbers
        upped_layer = cv2.pyrUp(gp[i+1]).astype(float)  # Gaussian blur and up-sample the next Gaussian pyramid layer
        laplacian = layer - upped_layer                 # L(i) = G(i) - expand(G(i+1))
        lp.append(laplacian)                            # append to pyramid
    lp.append(gp[-1])                                   # copy last layer of Gaussian pyramid
    return lp                                           # return lp


def main(base_img, patch_img, mask_img, output_img):
    """ Main program. Loads images and mask and performs the blending. Saves result to file.

    :param base_img: path to img
    :param patch_img: path to img
    :param mask_img: path to img
    :param output_img: path to img
    :return: void
    """
    # Load images
    base = cv2.imread(base_img, 0)
    patch = cv2.imread(patch_img, 0)
    mask = cv2.imread(mask_img, 0)

    # make Gaussian pyramids for all 3 images
    gp_base = make_gp(base)
    gp_patch = make_gp(patch)
    gp_mask = make_gp(mask)

    # make Laplacian pyramids for the base and patch images
    lp_base = make_lp(gp_base)
    lp_patch = make_lp(gp_patch)

    # create a combined Laplacian pyramid from the base and patch pyramids, masked by the mask pyramid
    lp_combined = []
    for i, mask_level in enumerate(gp_mask):
        # normalize the mask so it is in range [0, 1]
        lp_combined.append((mask_level/255) * lp_patch[i] + (1 - (mask_level/255)) * lp_base[i])

    # sum up all the layers of the combined pyramid to recover full image
    lp_combined.reverse()                                       # reverse the list. start at smallest image

    laplacian_sum = lp_combined[0]                              # initialize sum to 1st layer
    for layer in lp_combined[1:]:                               # for each layer, we up-sample the running sum
        laplacian_sum = cv2.pyrUp(laplacian_sum) + layer        # and add the new layer

    cv2.imwrite(output_img, laplacian_sum.astype(np.uint8))     # write the resulting image to file

    # display resulting image on screen
    plt.figure()
    plt.imshow(laplacian_sum, cmap='gray', vmin=0, vmax=255)
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == '__main__':
    base = '../images/2/dumbbells.png'
    patch = '../images/2/angry_trump.png'
    mask = '../images/2/mask.png'
    output = '../images/2/dumbbells-trump-blend.png'
    main(base, patch, mask, output)

    base = '../images/2/trump-base.png'
    patch = '../images/2/trump-patch_.png'
    mask = '../images/2/trump-mask.png'
    output = '../images/2/trump-blended.png'
    main(base, patch, mask, output)


