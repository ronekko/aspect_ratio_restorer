# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 20:54:58 2016

@author: yamane
"""

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
