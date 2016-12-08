# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 15:17:13 2016

@author: yamane
"""

import os
import sys
import numpy as np
import h5py
from fuel.datasets.hdf5 import H5PYDataset
from skimage import io, color, transform
import tqdm
import cv2


def create_path_list(data_location, dataset_root_dir):
    path_list = []
    root_dir_path = os.path.join(data_location, dataset_root_dir)

    for root, dirs, files in os.walk(root_dir_path):
        for file_name in tqdm.tqdm(files):
            file_path = os.path.join(root, file_name)
            image = io.imread(file_path)
            if len(image.shape) == 2:
                continue
            elif len(image[0][0]) != 3:
                continue
            path_list.append(file_path)
    return path_list


def output_path_list(path_list, output_root_dir):
    dirs = output_root_dir.split('\\')
    file_name = dirs[-1] + '.txt'
    output_root_dir = os.path.join(output_root_dir, file_name)

    f = open(output_root_dir, "w")
    for path in path_list:
        f.write(path + "\n")
    f.close()


def output_hdf5(path_list, data_chw, output_root_dir):
    num_data = len(path_list)

    channel, height, width = data_chw
    output_size = height

    dirs = output_root_dir.split('\\')
    file_name = dirs[-1] + '.hdf5'
    output_root_dir = os.path.join(output_root_dir, file_name)

    f = h5py.File(output_root_dir, mode='w')
    image_features = f.create_dataset('image_features',
                                      (num_data, height, width, channel),
                                      dtype='uint8')

    image_features.dims[0].label = 'batch'
    image_features.dims[1].label = 'height'
    image_features.dims[2].label = 'width'
    image_features.dims[3].label = 'channel'

    try:
        for i in tqdm.tqdm(range(num_data)):
            image = io.imread(path_list[i])
            image = crop_center(image)
            image = cv2.resize(image, (output_size, output_size))
            image = image.reshape(1, height, width, channel)
            image_features[i] = image

    except KeyboardInterrupt:
        print "割り込み停止が実行されました"

    f.flush()
    f.close()


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


def main(data_location, output_location, output_size):
    dataset_root_dir = r'Images'
    output_dir_name = 'output_size_' + str(output_size)
    output_root_dir = os.path.join(output_location, output_dir_name)
    data_chw = (3, output_size, output_size)

    if os.path.exists(output_root_dir):
        print u"すでに存在するため終了します."
        sys.exit()
    else:
        os.makedirs(output_root_dir)

    path_list = create_path_list(data_location, dataset_root_dir)
    shuffled_path_list = np.random.permutation(path_list)
    output_path_list(shuffled_path_list, output_root_dir)
    output_hdf5(shuffled_path_list, data_chw, output_root_dir)


if __name__ == '__main__':
    data_location = r'E:\stanford_Dogs_Dataset'
    output_location = r'E:\stanford_Dogs_Dataset\raw_dataset'
    output_size = 500

    main(data_location, output_location, output_size)
