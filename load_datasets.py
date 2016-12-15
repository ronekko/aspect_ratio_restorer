# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:06:51 2016

@author: yamane
"""

import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue

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


def load_toy_stream(randomcirclesquaredataset, batch_size):
    """Parameters
    ----------
    randomcirclesquaredataset : RandomCircleSqureDatasetクラスインスタンス.

    batch_size : バッチサイズ"""
    dataset = IterableDataset(
        batch_images(randomcirclesquaredataset, batch_size))

    stream_train = DataStream(dataset)
    stream_test = DataStream(dataset)

    return stream_train, stream_test


def batch_images(randomcirclesquaredataset, batch_size):
    while True:
        images = []
        for i in range(batch_size):
            image = randomcirclesquaredataset.create_image()
            images.append(image)
        batch = np.stack(images, axis=0)
        yield batch


def dog_minibatch(queue, it, aspect_ratio_max, aspect_ratio_min,
                  output_size, crop_size):
    X_batch = it.next()[0]
    images = []
    ts = []
    for b in range(X_batch.shape[0]):
        image = X_batch[b]
        r = utility.sample_random_aspect_ratio(aspect_ratio_max,
                                               aspect_ratio_min)
        image = utility.change_aspect_ratio(image, r)
        square_image = utility.crop_center(image)
        resize_image = cv2.resize(
            square_image, (output_size, output_size))
        resize_image = utility.random_crop_and_flip(resize_image,
                                                    crop_size)
        images.append(resize_image)
        t = np.log(r)
        ts.append(t)
    X = np.stack(images, axis=0)
    X = np.transpose(X, (0, 3, 1, 2))
    X = X.astype(np.float32)
    T = np.array(ts, dtype=np.float32).reshape(-1, 1)
    queue.put((X, T))


if __name__ == '__main__':
    hdf5_filepath = r'E:\stanford_Dogs_Dataset\raw_dataset_binary\output_size_500\output_size_500.hdf5'
    batch_size = 100
    p = [0.3, 0.3, 0.4]  # [円の生成率、四角の生成率、円と四角の混合生成率]
    aspect_ratio_max = 2.5
    aspect_ratio_min = 1.0
    output_size = 256
    crop_size = 224

    draw_toy_image_class = RandomCircleSquareDataset(p=p)

    dog_stream_train, dog_stream_test = load_dog_stream(
        hdf5_filepath, batch_size)
    toy_stream_train, toy_stream_test = load_toy_stream(
        draw_toy_image_class, batch_size)
    dog_it_train = dog_stream_train.get_epoch_iterator()
    queue_train = Queue(10)
    process_train = Process(target=dog_minibatch,
                            args=(queue_train, dog_it_train, aspect_ratio_max,
                                  aspect_ratio_min, output_size, crop_size))
    process_train.start()
    for i in range(10):
        print i
        X, T = queue_train.get()
        image = np.transpose(X, (0, 2, 3, 1))
        plt.imshow(image[0]/256.0)
        plt.show()
