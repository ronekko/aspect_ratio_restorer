# -*- coding: utf-8 -*-
"""
Created on Fri Dec 09 19:50:48 2016

@author: yamane
"""

import toydata
import numpy as np
import h5py
import cv2
from skimage import color


def create_minibatch(file_path, data, batch_size=100, min_ratio=1,
                     max_ratio=4, crop_size=224, output_size=256):
    dataset = h5py.File(file_path)
    image_features = dataset['image_features']
    r_min = min_ratio
    r_max = max_ratio

    num_batches = len(data) / batch_size

    while True:
        for indexes in np.array_split(data, num_batches):
            images = []
            ts = []
            image_batch = image_features[indexes.tolist()]
            for i in range(len(indexes)):
                image = image_batch[i]
                r = toydata.sample_random_aspect_ratio(r_max, r_min)
                image = toydata.change_aspect_ratio(image, r)
                square_image = toydata.crop_center(image)
                resize_image = cv2.resize(square_image,
                                          (output_size, output_size))
                resize_image = toydata.random_crop_and_flip(resize_image,
                                                            crop_size)
                images.append(resize_image)
                t = np.log(r)
                ts.append(t)
            X = np.stack(images, axis=0)
            X = np.transpose(X, (0, 3, 1, 2))
            X = X.astype(np.float32)
            X = X / 256.0
            T = np.array(ts, dtype=np.float32).reshape(-1, 1)
            return X, T


def change_gray_minibatch(X):
    images = []
    batch_size = X.shape[0]
    h, w = X.shape[2:]
    for b in range(batch_size):
        X_hwc = np.transpose(X[b], (1, 2, 0))
        X_gray = color.rgb2gray(X_hwc)
        X_gray = X_gray.reshape((1, h, w))
        images.append(X_gray)
    X = np.stack(images, axis=0)
    X = X.astype(np.float32)
    return X


if __name__ == '__main__':
    # 超パラメータ
    max_iteration = 500  # 繰り返し回数
    batch_size = 1
    num_train = 20000
    num_test = 100
    output_size = 256
    crop_size = 224
    aspect_ratio_max = 3
    aspect_ratio_min = 1.0
    file_path = r'E:\stanford_Dogs_Dataset\raw_dataset_binary\output_size_500\output_size_500.hdf5'
    test_data = range(num_train, num_train + num_test)
    X, T = create_minibatch(file_path, test_data, batch_size, aspect_ratio_min,
                            aspect_ratio_max, crop_size, output_size)