# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 15:02:07 2016

@author: yamane
"""

import os
import sys
import numpy as np
import h5py
from fuel.datasets.hdf5 import H5PYDataset
from skimage import io, color, transform
import tqdm


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


def random_aspect_ratio_and_square_image(image, min_ratio, max_ratio):
    h_image, w_image = image.shape[:2]
    T = 0

    while True:
        aspect_ratio = np.random.rand() * 1.5  # 0.5~2の乱数を生成
        if aspect_ratio >= min_ratio and aspect_ratio < 1.0:
            break
        elif aspect_ratio > 1.0 and aspect_ratio <= max_ratio:
            break

    square_image = transform.resize(image, (224, 224))

    if np.random.rand() > 0.5:  # 半々の確率で
        w_image = w_image * aspect_ratio
        T = 1

    if h_image >= w_image:
        h_image = int(h_image * (224.0 / w_image))
        if (h_image % 2) == 1:
            h_image = h_image + 1
        w_image = 224
        resize_image = transform.resize(image, (h_image, w_image))
        diff = h_image - w_image
        margin = int(diff / 2)
        if margin == 0:
            square_image = resize_image
        else:
            square_image = resize_image[margin:-margin, :]
    else:
        w_image = int(w_image * (224.0 / h_image))
        if (w_image % 2) == 1:
            w_image = w_image + 1
        h_image = 224
        resize_image = transform.resize(image, (h_image, w_image))
        diff = w_image - h_image
        margin = int(diff / 2)
        if margin == 0:
            square_image = resize_image
        else:
            square_image = resize_image[:, margin:-margin]

    return square_image, T


def output_hdf5(path_list, data_chw, output_root_dir, min_ratio, max_ratio):
    num_data = len(path_list)

    channel, height, width = data_chw

    dirs = output_root_dir.split('\\')
    file_name = dirs[-1] + '.hdf5'
    output_root_dir = os.path.join(output_root_dir, file_name)

    f = h5py.File(output_root_dir, mode='w')
    image_features = f.create_dataset('image_features',
                                      (num_data, channel, height, width),
                                      dtype='uint8')
    targets = f.create_dataset('targets', (num_data, 1), dtype='uint8')

    image_features.dims[0].label = 'batch'
    image_features.dims[1].label = 'height'
    image_features.dims[2].label = 'width'
    image_features.dims[3].label = 'channel'
    targets.dims[0].label = 'batch'

    try:
        for i in tqdm.tqdm(range(num_data)):
            image = io.imread(path_list[i])
            image, t = random_aspect_ratio_and_square_image(
                    image, min_ratio, max_ratio)
            image = np.transpose(image, (2, 0, 1))
            image = np.reshape(image, (1, channel, height, width))
            image = image * 256
            image_features[i] = image
            targets[i] = t

    except KeyboardInterrupt:
        print "割り込み停止が実行されました"

    f.flush()
    f.close()


def main(data_location, output_location, output_size, min_ratio, max_ratio):
    dataset_root_dir = r'stanford_Dogs_Dataset\Images'
    output_dir_name = 'max_aspect_ratio_' + str(max_ratio)
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
    output_hdf5(shuffled_path_list, data_chw, output_root_dir, min_ratio, max_ratio)


if __name__ == '__main__':
    data_location = r'E:'
    output_location = r'E:\stanford_Dogs_Dataset\raw_dataset_binary'
    output_size = 224
    min_ratio = 2.0 / 3.0
    max_ratio = 3.0 / 2.0

    main(data_location, output_location, output_size, min_ratio, max_ratio)
