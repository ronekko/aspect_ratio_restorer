# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:06:51 2016

@author: yamane
"""

import numpy as np
import h5py
from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.datasets import IterableDataset
import utility
from datasets import RandomCircleSquareDataset


def load_dog_stream(hdf5_filepath, batch_size, train_size=20000,
                    shuffle=False):
    indices_train = range(0, train_size)
    indices_test = range(train_size, 20579)

    h5py_file = h5py.File(hdf5_filepath)
    num_examples = len(h5py_file['image_features'])
    split_train = (0, num_examples)
    split_dict = dict(train=dict(image_features=split_train))
    h5py_file.attrs["split"] = H5PYDataset.create_split_array(split_dict)
    dataset = H5PYDataset(h5py_file, ['train'])

    scheme_class = ShuffledScheme if shuffle else SequentialScheme
    scheme_train = scheme_class(indices_train, batch_size=batch_size)
    scheme_test = scheme_class(indices_test, batch_size=batch_size)

    stream_train = DataStream(dataset, iteration_scheme=scheme_train)
    stream_test = DataStream(dataset, iteration_scheme=scheme_test)

    return stream_train, stream_test


def load_toy_stream(draw_toy_image_class, batch_size):

    dataset = IterableDataset(batch_images(draw_toy_image_class, batch_size))

    stream_train = DataStream(dataset)
    stream_test = DataStream(dataset)

    return stream_train, stream_test


def batch_images(draw_toy_image_class, batch_size):
    while True:
        images = []
        for i in range(batch_size):
            image = draw_toy_image_class.create_image()
            images.append(image)
        batch = np.stack(images, axis=0)
        yield batch


if __name__ == '__main__':
    hdf5_filepath = r'E:\stanford_Dogs_Dataset\raw_dataset_binary\output_size_500\output_size_500.hdf5'
    batch_size = 100

    draw_toy_image_class = RandomCircleSquareDataset()

    dog_stream_train, dog_stream_test = load_dog_stream(
        hdf5_filepath, batch_size)
    toy_stream_train, toy_stream_test = load_toy_stream(
        draw_toy_image_class, batch_size)
