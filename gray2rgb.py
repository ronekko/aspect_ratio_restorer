# -*- coding: utf-8 -*-
"""
Created on Fri Dec 09 23:43:22 2016

@author: yamane
"""

import toydata
import numpy as np
from skimage import color


def create_minibatch(batch_size):
    image_size = 500
    circle_r_min = 50
    circle_r_max = 150
    size_min = 199
    size_max = 200
    p = [0, 1, 0]
    output_size = 224
    aspect_ratio_max = 3.0
    aspect_ratio_min = 1.0

    dataset = toydata.RandomCircleSquareDataset(
        image_size, circle_r_min, circle_r_max, size_min, size_max, p,
        output_size, aspect_ratio_max, aspect_ratio_min)

    X, T = dataset.minibatch_regression(batch_size)
    return X, T


def change_rgb_minibatch(X):
    images = []
    batch_size = X.shape[0]
    h, w = X.shape[2:]
    for b in range(batch_size):
        X_gray = color.gray2rgb(X[b])
        X_gray = np.transpose(X_gray, (0, 3, 1, 2))
        X_gray = X_gray.reshape((3, h, w))
        images.append(X_gray)
    X = np.stack(images, axis=0)
    X = X.astype(np.float32)
    return X


if __name__ == '__main__':
    batch_size = 100
    X_test, T_test = create_minibatch(batch_size)
    X_test = change_rgb_minibatch(X_test)
