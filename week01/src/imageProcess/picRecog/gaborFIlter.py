import os
import cv2
import numpy as np

img = cv2.imread("img.png")
b, g, r = cv2.split(img)


def gabor_fn(img_data, sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # ------这部分内容是为了确定卷积核的大小------
    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    x = img_data[:, 0]
    y = img_data[:, 1]
    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * \
         np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb


result = gabor_fn(g, 1, 0.3, 2, 50*np.pi/180., 0.3)
print(result)


def gaborFilter(lambda_, theta, sigma2, gamma, psi):
    sigma_x = sigma2
