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
from skimage import io
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


def output_hdf5(path_list, output_root_dir):
    num_data = len(path_list)
    shapes = []

    dirs = output_root_dir.split('\\')
    file_name = dirs[-1] + '.hdf5'
    output_root_dir = os.path.join(output_root_dir, file_name)

    f = h5py.File(output_root_dir, mode='w')
    dtype = h5py.special_dtype(vlen=np.dtype('uint8'))
    image_features = f.create_dataset('image_features',
                                      (num_data,),
                                      dtype=dtype)

    image_features.dims[0].label = 'batch'

    try:
        for i in tqdm.tqdm(range(num_data)):
            image = io.imread(path_list[i])
            shapes.append(image.shape)
            image_features[i] = image.flatten()

        shapes = np.array(shapes).astype(np.int32)
        image_features_shapes = f.create_dataset('image_features_shapes',
                                                 (num_data, 3),
                                                 dtype=np.int32)
        image_features_shapes[...] = shapes

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

        # specify the splits
        split_train = (0, num_data)
        split_dict = dict(train=dict(image_features=split_train))
        f.attrs["split"] = H5PYDataset.create_split_array(split_dict)

    except KeyboardInterrupt:
        print "割り込み停止が実行されました"

    f.flush()
    f.close()


def main(data_location, output_location):
    dataset_root_dir = r'VOCdevkit\VOC2012\JPEGImages'
    output_root_dir = output_location

    if os.path.exists(output_root_dir):
        print u"すでに存在するため終了します."
        sys.exit()
    else:
        os.makedirs(output_root_dir)

    path_list = create_path_list(data_location, dataset_root_dir)
    shuffled_path_list = np.random.permutation(path_list)
    output_path_list(shuffled_path_list, output_root_dir)
    output_hdf5(shuffled_path_list, output_root_dir)


if __name__ == '__main__':
    # PASCALVOC2012データセットの保存場所
    data_location = r'E:\voc'
    # hdf５ファイルを保存する場所
    output_location = r'E:\voc\hdf5_dataset_c'

    main(data_location, output_location)
