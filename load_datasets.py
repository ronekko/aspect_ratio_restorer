# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:06:51 2016

@author: yamane
"""

import os
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


def load_dog_stream(hdf5_filepath, batch_size, train_size=16500,
                    validation_size=500, test_size=100, shuffle=False):
    valid_size = train_size + validation_size
    test_size = valid_size + test_size
    indices_train = range(0, train_size)
    indices_valid = range(train_size, valid_size)
    indices_test = range(valid_size, test_size)

    h5py_file = h5py.File(hdf5_filepath)
    dataset = H5PYDataset(h5py_file, ['train'])

    scheme_class = ShuffledScheme if shuffle else SequentialScheme
    scheme_train = scheme_class(indices_train, batch_size=batch_size)
    scheme_valid = scheme_class(indices_valid, batch_size=batch_size)
    scheme_test = scheme_class(indices_test, batch_size=batch_size)

    stream_train = DataStream(dataset, iteration_scheme=scheme_train)
    stream_valid = DataStream(dataset, iteration_scheme=scheme_valid)
    stream_test = DataStream(dataset, iteration_scheme=scheme_test)
    stream_train.get_epoch_iterator().next()
    stream_valid.get_epoch_iterator().next()
    stream_test.get_epoch_iterator().next()

    return stream_train, stream_valid, stream_test


def load_toy_stream(batch_size, ):
    """Parameters
    ----------
    randomcirclesquaredataset : RandomCircleSqureDatasetクラスインスタンス.
    batch_size : バッチサイズ"""
    dataset = IterableDataset(RandomCircleSquareDataset(batch_size=batch_size))

    stream_train = DataStream(dataset)
    stream_valid = DataStream(dataset)
    stream_test = DataStream(dataset)
    stream_train.get_epoch_iterator().next()
    stream_valid.get_epoch_iterator().next()
    stream_test.get_epoch_iterator().next()

    return stream_train, stream_valid, stream_test


def data_crop(X_batch, aspect_ratio_max=3.0, output_size=256, crop_size=224,
              random=True, t=0):
    images = []
    ts = []

    for b in range(X_batch.shape[0]):
        # 補間方法を乱数で設定
        u = np.random.randint(5)
        image = X_batch[b]
        if random is False:
            t = t
        else:
            t = utility.sample_random_aspect_ratio(np.log(aspect_ratio_max),
                                                   -np.log(aspect_ratio_max))
        r = np.exp(t)
        # 歪み画像生成
        image = utility.change_aspect_ratio(image, r, u)
        # 中心切り抜き
        square_image = utility.crop_center(image)
        if u == 0:
            resize_image = cv2.resize(square_image, (output_size, output_size),
                                      interpolation=cv2.INTER_NEAREST)
        elif u == 1:
            resize_image = cv2.resize(square_image, (output_size, output_size),
                                      interpolation=cv2.INTER_LINEAR)
        elif u == 2:
            resize_image = cv2.resize(square_image, (output_size, output_size),
                                      interpolation=cv2.INTER_AREA)
        elif u == 3:
            resize_image = cv2.resize(square_image, (output_size, output_size),
                                      interpolation=cv2.INTER_CUBIC)
        elif u == 4:
            resize_image = cv2.resize(square_image, (output_size, output_size),
                                      interpolation=cv2.INTER_LANCZOS4)
        if random is False:
            crop_image = utility.crop_224(resize_image)
        else:
            crop_image = utility.random_crop_and_flip(resize_image, crop_size)
        images.append(crop_image)
        ts.append(t)
    X = np.stack(images, axis=0)
    X = np.transpose(X, (0, 3, 1, 2))
    X = X.astype(np.float32)
    T = np.array(ts, dtype=np.float32).reshape(-1, 1)
    return X, T


def data_padding(X_batch, aspect_ratio_max=3.0, output_size=256, crop_size=224,
                 random=True, t=0):
    images = []
    ts = []
    for b in range(X_batch.shape[0]):
        # 補間方法を乱数で設定
        u = np.random.randint(5)
        image = X_batch[b]
        if random is False:
            t = t
        else:
            t = utility.sample_random_aspect_ratio(np.log(aspect_ratio_max),
                                                   -np.log(aspect_ratio_max))
        r = np.exp(t)
        image = utility.change_aspect_ratio(image, r, u)
        square_image = utility.crop_center(image)
        if u == 0:
            resize_image = cv2.resize(square_image, (output_size, output_size),
                                      interpolation=cv2.INTER_NEAREST)
        elif u == 1:
            resize_image = cv2.resize(square_image, (output_size, output_size),
                                      interpolation=cv2.INTER_LINEAR)
        elif u == 2:
            resize_image = cv2.resize(square_image, (output_size, output_size),
                                      interpolation=cv2.INTER_AREA)
        elif u == 3:
            resize_image = cv2.resize(square_image, (output_size, output_size),
                                      interpolation=cv2.INTER_CUBIC)
        elif u == 4:
            resize_image = cv2.resize(square_image, (output_size, output_size),
                                      interpolation=cv2.INTER_LANCZOS4)
        resize_image = resize_image[..., None]
        if random is False:
            crop_image = utility.crop_224(resize_image)
        else:
            crop_image = utility.random_crop_and_flip(resize_image, crop_size)
        images.append(crop_image)
        ts.append(t)
    X = np.stack(images, axis=0)
    X = np.transpose(X, (0, 3, 1, 2))
    X = X.astype(np.float32)
    T = np.array(ts, dtype=np.float32).reshape(-1, 1)
    return X, T


def load_data(queue, stream, crop, aspect_ratio_max=3.0, output_size=256,
              crop_size=224, random=True, t=0):
    while True:
        try:
            for X in stream.get_epoch_iterator():
                if crop is True:
                    X, T = data_crop(X[0], aspect_ratio_max, output_size,
                                     crop_size, random, t)
                else:
                    X, T = data_padding(X[0], aspect_ratio_max, crop_size,
                                        random, t)
                queue.put((X, T), timeout=0.05)
        except Full:
            print 'Full'


if __name__ == '__main__':
#    hdf5_filepath = r'E:\voc\raw_dataset\output_size_256\output_size_256.hdf5'
#    hdf5_filepath = r'E:\voc2012\raw_dataset\output_size_500\output_size_500.hdf5'
    hdf5_filepath = r'E:\voc\variable_dataset\output_size_256\output_size_256.hdf5'  # データセットファイル保存場所
    assert os.path.exists(hdf5_filepath)
    batch_size = 100
    p = [0.3, 0.3, 0.4]  # [円の生成率、四角の生成率、円と四角の混合生成率]
    aspect_ratio_max = 3.0
    output_size = 256
    crop_size = 224
    crop = True

#    draw_toy_image_class = RandomCircleSquareDataset(p=p)

    dog_stream_train, dog_stream_valid, dog_stream_test = load_dog_stream(
        hdf5_filepath, batch_size)
#    toy_stream_train, toy_stream_test = load_toy_stream(batch_size)

#    for batch in dog_stream_train.get_epoch_iterator():
#        plt.imshow(batch[0][0])
#        plt.show()

    q_dog_train = Queue(10)
    process_dog = Process(target=load_data,
                          args=(q_dog_train, dog_stream_train, crop))

    process_dog.start()

#    q_toy_train = Queue(10)
#    process_toy = Process(target=load_data,
#                          args=(q_toy_train, toy_stream_train, False))
#    process_toy.start()

    for i in range(10):
        X, T = q_dog_train.get()
        image = np.transpose(X, (0, 2, 3, 1))
        plt.imshow(image[0]/256.0)
        plt.show()
    process_dog.terminate()

#    for i in range(10):
#        X, T = q_toy_train.get()
#        plt.imshow(X[0][0])
#        plt.show()
#    process_toy.terminate()
