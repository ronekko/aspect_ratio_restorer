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
from Queue import Full

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
    stream_train.get_epoch_iterator().next()
    stream_test.get_epoch_iterator().next()

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
    stream_train.get_epoch_iterator().next()
    stream_test.get_epoch_iterator().next()

    return stream_train, stream_test


def batch_images(randomcirclesquaredataset, batch_size):
    while True:
        images = []
        for i in range(batch_size):
            image = randomcirclesquaredataset.create_image()
            images.append(image)
        batch = np.stack(images, axis=0)
        yield batch


def data_crop(X_batch, aspect_ratio_max=2.5, aspect_ratio_min=1,
              output_size=256, crop_size=224):
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
    return X, T


def data_padding(X_batch, aspect_ratio_max=2.5, aspect_ratio_min=1,
                 output_size=224):
    images = []
    ts = []
    for b in range(X_batch.shape[0]):
        image = X_batch[b]
        r = utility.sample_random_aspect_ratio(aspect_ratio_max,
                                               aspect_ratio_min)
        image = utility.change_aspect_ratio(image, r)
        square_image = utility.padding_image(image)
        # cv2.resize:(image, (w, h))
        # transform.resize:(image, (h, w))
        resize_image = cv2.resize(
            square_image, (output_size, output_size),
            interpolation=cv2.INTER_NEAREST)
        resize_image = resize_image[..., None]
        images.append(resize_image)
        t = np.log(r)
        ts.append(t)
    X = np.stack(images, axis=0)
    X = np.transpose(X, (0, 3, 1, 2))
    X = X.astype(np.float32)
    T = np.array(ts, dtype=np.float32).reshape(-1, 1)
    return X, T


def load_data(queue, stream, crop):
    while True:
        try:
            for X in stream.get_epoch_iterator():
                if crop is True:
                    X, T = data_crop(X[0])
                else:
                    X, T = data_padding(X[0])
                queue.put((X, T), timeout=0.05)
        except Full:
            print 'Full'


if __name__ == '__main__':
    hdf5_filepath = r'E:\stanford_Dogs_Dataset\raw_dataset_binary\output_size_500\output_size_500.hdf5'
    batch_size = 100
    p = [0.3, 0.3, 0.4]  # [円の生成率、四角の生成率、円と四角の混合生成率]
    aspect_ratio_max = 1.5
    aspect_ratio_min = 1.0
    output_size = 256
    crop_size = 224
    crop = True

    draw_toy_image_class = RandomCircleSquareDataset(p=p)

    dog_stream_train, dog_stream_test = load_dog_stream(
        hdf5_filepath, batch_size)
    toy_stream_train, toy_stream_test = load_toy_stream(
        draw_toy_image_class, batch_size)

#    q_dog_train = Queue(10)
    q_toy_train = Queue(10)
#    process_dog = Process(target=load_data,
#                          args=(q_dog_train, dog_stream_train, crop))
    process_toy = Process(target=load_data,
                          args=(q_toy_train, toy_stream_train, crop))
#    process_dog.start()
    process_toy.start()
#    load_data(q_toy_train, toy_stream_train, padding)
    for i in range(10):
        if crop is True:
            1
#            X, T = q_dog_train.get()
#            image = np.transpose(X, (0, 2, 3, 1))
#            plt.imshow(image[0]/256.0)
#            plt.show()
        else:
            1
            X, T = q_toy_train.get()
            plt.imshow(X[0][0])
            plt.show()
#    process_dog.terminate()
    process_toy.terminate()
