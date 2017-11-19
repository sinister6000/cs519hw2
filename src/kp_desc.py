# Allen Kim
# CS 519
# Assignment 2, #4


import cv2
import matplotlib.pyplot as plt
import numpy as np


def match_SIFT(im1, im2, out_img):
    """ Performs image matching on SIFT keypoints and descriptors.

    :return: void
    """
    img1 = cv2.imread(im1, 0)           # queryImage
    img2 = cv2.imread(im2, 0)           # trainImage
    gray1 = img1.copy()
    gray2 = img2.copy()

    sift = cv2.xfeatures2d.SIFT_create()                    # Initiate SIFT detector
    kp1, des1 = sift.detectAndCompute(gray1, None, img1)    # find the keypoints and descriptors with SIFT
    kp2, des2 = sift.detectAndCompute(gray2, None, img2)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)        # create BFMatcher object
    matches = bf.match(des1, des2)                          # Match descriptors

    matches = sorted(matches, key=lambda x: x.distance)     # Sort according to distance

    # Create storage for output of drawMatches
    match_img = np.zeros((img1.shape[0], img1.shape[1] + img2.shape[1], 3), np.uint8)
    # Draw first 30 matches
    disp_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], match_img, matchColor=(0,255,255), flags=2)

    cv2.imwrite(out_img, disp_img)              # output match image to file
    plt.figure()                                # display match image
    plt.xticks([])
    plt.yticks([])
    plt.imshow(disp_img)
    plt.show()


def match_ORB(im1, im2, out_img):
    """ Performs image matching on ORB keypoints and descriptors.

    :return: void
    """
    # read images
    img1 = cv2.imread(im1, 0)           # queryImage
    img2 = cv2.imread(im2, 0)           # trainImage

    orb = cv2.ORB_create()              # Initiate ORB detector
    kp1, des1 = orb.detectAndCompute(img1, None)            # find the keypoints and descriptors with ORB
    kp2, des2 = orb.detectAndCompute(img2, None)            # find the keypoints and descriptors with ORB

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)       # create BFMatcher object, NORM_HAMMING for ORB
    matches = bf.match(des1, des2)                              # Match descriptors
    matches = sorted(matches, key=lambda x: x.distance)         # Sort according to distance

    # Create storage for output of drawMatches
    match_img = np.zeros((img1.shape[0], img1.shape[1] + img2.shape[1], 3), np.uint8)
    # Draw first 30 matches
    disp_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], match_img, matchColor=(255,100,100), flags=2)

    cv2.imwrite(out_img, disp_img)          # output match image to file
    plt.figure()                            # display match image
    plt.xticks([])
    plt.yticks([])
    plt.imshow(disp_img)
    plt.show()


if __name__ == '__main__':
    match_SIFT('../images/4/chicago.jpg', '../images/4/chicago2.jpg', '../images/4/chicago_match_SIFT.png')
    match_ORB('../images/4/chicago.jpg', '../images/4/chicago2.jpg', '../images/4/chicago_match_ORB.png')
