import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np


def main():

    im = cv2.imread('../images/1/pool table.jpg', 0)
    im_med5 = cv2.medianBlur(im,5)
    # im_med5 = cv2.GaussianBlur(im, (5,5), 0)


    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    # Gradient-X
    grad_x = cv2.Sobel(im_med5, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # grad_x = cv2.Scharr(im_med5,ddepth,1,0)

    # Gradient-Y
    grad_y = cv2.Sobel(im_med5, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # grad_y = cv2.Scharr(im_med5,ddepth,0,1)

    abs_grad_x = cv2.convertScaleAbs(grad_x)  # converting back to uint8
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    dst = cv2.add(abs_grad_x,abs_grad_y)

    dst = cv2.GaussianBlur(dst, (9,9), 0, dst)
    p3 = plt.figure(3)
    plt.imshow(dst)

    im_can = cv2.Canny(dst, 50, 150, True)
    minLineLength = 1
    lines = cv2.HoughLines(im_can, 100, np.pi/180, 100)

    x0=0
    xend = im.shape[1]-1

    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)

        try:
            y0 = np.round((-a/b) * x0 + rho/b, 0)
            y0 = int(y0)
            yend = np.round((-a/b)*xend + rho/b, 0)
            yend = int(yend)
            cv2.line(im, (x0, y0), (xend, yend),(225,0,0),2)
        except ValueError:
            pass

    cv2.imshow("img", im)
    cv2.waitKey()
    # for x1, y1, x2, y2 in lines[0]:
    #     cv2.line(im, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite('../images/1/houghlines5.jpg', im)

    # im_med5 = cv2.medianBlur(im, 9)
    # im_can = cv2.Sobel(im)

    p1 = plt.figure(1)
    plt.imshow(dst)

    p2 = plt.figure(2)
    plt.imshow(im_can)
    plt.show(block=True)




if __name__ == '__main__':
    main()
