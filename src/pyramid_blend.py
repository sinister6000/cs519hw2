# Allen Kim
# CS 519
# Assignment 2, #2


import cv2
import matplotlib.pyplot as plt


def make_gp(src):
    """ Creates Gaussian pyramid based on image src

    :param src: source image
    :return: [images] that make up the Gaussian pyramid.
             List is ordered from largest to smallest images.
    """
    src_ = src.copy()                   # make a copy of source image
    pyramid = [src_]                    # 1st element of pyramid is image itself
    for _ in range(8):                  # pyramid will contain 9 levels
        src_ = cv2.pyrDown(src_)        # Gaussian blur and downsample
        pyramid.append(src_)            # append to pyramid
    return pyramid                      # return pyramid


def make_lp(gp):
    """ Creates Laplacian pyramid based on a Gaussian pyramid.

    :param gp: [images] forming a Gaussian pyramid
    :return: [images] that form the corresponding Laplacian pyramid
    """
    lp = []                                             # initialize list that we will return
    for i, level in enumerate(gp[:8]):                  # use enumerate to get index and element from Gaussian pyramid
        level = level.astype(float)                     # convert to float to allow for negative numbers
        upped_level = cv2.pyrUp(gp[i+1]).astype(float)  # Gaussian blur and upsample the next Gaussian pyramid layer
        laplacian = level - upped_level                 # L(i) = G(i) - expand(G(i+1))
        lp.append(laplacian)                            # append to pyramid
    lp.append(gp[8])                                    # copy last level of Gaussian pyramid
    return lp                                           # return lp


def main():
    """ Main program. Loads images and mask and performs the blending. Saves result to file.

    :return: void
    """
    # Load images
    base = cv2.imread('../images/2/trump-base.png', 0)
    patch = cv2.imread('../images/2/trump-patch_.png', 0)
    mask = cv2.imread('../images/2/trump-mask.png', 0)

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
        # we normalize the mask so it is in range [0, 1]
        lp_combined.append((mask_level/255) * lp_patch[i] + (1 - (mask_level/255)) * lp_base[i])

    # sum up all the layers of the combined pyramid to recover full image
    lp_combined.reverse()                                       # reverse the list. start at smallest image
    laplacian_sum = lp_combined[0]
    for layer in lp_combined[1:]:                               # for each layer, we upsample the running sum
        laplacian_sum = cv2.pyrUp(laplacian_sum) + layer        # and add the new layer

    cv2.imwrite('../images/2/trump-blended.png', laplacian_sum)      # write the resulting image to file

    # display resulting image on screen
    plt.figure()
    plt.imshow(laplacian_sum, cmap='gray', vmin=0, vmax=255)
    plt.show()


if __name__ == '__main__':
    print("I know it's technically not two different images that I'm blending, but the effect is pretty funny.")
    main()
