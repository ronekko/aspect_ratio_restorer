# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:17:49 2017

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
import utility


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

    dirs = output_root_dir.split('\\')
    file_name = dirs[-1] + '.hdf5'
    output_root_dir = os.path.join(output_root_dir, file_name)

    f = h5py.File(output_root_dir, mode='w')
    dtype = h5py.special_dtype(vlen=np.dtype('uint8'))
    image_features = f.create_dataset('image_features',
                                      (num_data,),
                                      dtype=dtype)
    image_features_shapes = f.create_dataset('image_features_shapes',
                                             (num_data, 3),
                                             dtype='uint8')

    image_features.dims[0].label = 'batch'

    try:
        for i in tqdm.tqdm(range(num_data)):
            image = io.imread(path_list[i])
            image_shape = image.shape
            image_features[i] = image.flatten()
            image_features_shapes[i] = image_shape

        image_features.dims.create_scale(image_features_shapes, 'shapes')
        image_features.dims[0].attach_scale(image_features_shapes)

        image_features_shape_labels = f.create_dataset(
            'image_features_shape_labels', (3,), dtype='S7')
        image_features_shape_labels[...] = [
             'height'.encode('utf8'), 'width'.encode('utf8'),
             'channel'.encode('utf8')]
        image_features.dims.create_scale(
            image_features_shape_labels, 'shape_labels')
        image_features.dims[0].attach_scale(image_features_shape_labels)

    except KeyboardInterrupt:
        print "割り込み停止が実行されました"

    f.flush()
    f.close()


def main(data_location, output_location, output_size):
    dataset_root_dir = r'E:\voc\VOCdevkit\VOC2012\JPEGImages'
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
    data_location = r'E:\voc'
    output_location = r'E:\voc\variable_dataset'
    output_size = 256

    main(data_location, output_location, output_size)
