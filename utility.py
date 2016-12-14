# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 20:54:58 2016

@author: yamane
"""

import numpy as np
import cv2


def change_aspect_ratio(image, aspect_ratio):
    h_image, w_image = image.shape[:2]
    r = aspect_ratio

    if r == 1:
        return image
    elif r > 1:
        w_image = int(w_image * r)
    else:
        h_image = int(h_image / float(r))
    # cv2.resize:（image, (w, h))
    # transform.resize:(image, (h, w))
    if image.shape[2] == 1:
        resize_image = cv2.resize(image, (w_image, h_image))[..., None]
    else:
        resize_image = cv2.resize(image, (w_image, h_image))

    return resize_image


def crop_center(image):
    height, width = image.shape[:2]
    left = 0
    right = width
    top = 0
    bottom = height

    if height >= width:  # 縦長の場合
        output_size = width
        margin = int((height - width) / 2)
        top = margin
        bottom = top + output_size
    else:  # 横長の場合
        output_size = height
        margin = int((width - height) / 2)
        left = margin
        right = left + output_size

    square_image = image[top:bottom, left:right]

    return square_image


def padding_image(image):
    height, width = image.shape[:2]
    left = 0
    right = width
    top = 0
    bottom = height

    if height >= width:  # 縦長の場合
        output_size = height
        margin = int((height - width) / 2)
        left = margin
        right = left + width
    else:  # 横長の場合
        output_size = width
        margin = int((width - height) / 2)
        top = margin
        bottom = top + height

    square_image = np.zeros((output_size, output_size, 1))
    square_image[top:bottom, left:right] = image

    return square_image
