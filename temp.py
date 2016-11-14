# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:12:33 2016

@author: yamane
"""


import numpy as np
from chainer import cuda, optimizers, Chain, serializers
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
from skimage import io, transform
import time
import copy
import tqdm


def random_aspect_ratio_and_resize_image(image):
    h_image, w_image = image.shape[:2]

    aspect_ratio = np.random.rand() * 2  # 0~2の乱数を生成
    while True:
        aspect_ratio = np.random.rand() * 2
        if aspect_ratio > 0:
            break

    w_image = w_image * aspect_ratio

    square_image = transform.resize(image, (500, 500))

    if h_image >= w_image:
        w_image = int(w_image * (500.0 / h_image))
        if (w_image % 2) == 1:
            w_image = w_image + 1
        h_image = 500
        resize_image = transform.resize(image, (h_image, w_image))
        diff = h_image - w_image
        margin = int(diff / 2)
        square_image[:, :margin] = 0
        square_image[:, margin:-margin] = resize_image
        square_image[:, -margin:] = 0
    else:
        h_image = int(h_image * (500.0 / w_image))
        if (h_image % 2) == 1:
            h_image = h_image + 1
        w_image = 500
        resize_image = transform.resize(image, (h_image, w_image))
        diff = w_image - h_image
        margin = int(diff / 2)
        square_image[:margin, :] = 0
        square_image[margin:-margin, :] = resize_image
        square_image[-margin:, :] = 0

    if np.random.rand() > 0.5:  # 半々の確率で
        resize_image = resize_image[:, ::-1]  # 左右反転

    return square_image, aspect_ratio


def read_images_and_T(image_list, indexes):
    images = []
    ratios = []
    T = []

    for i in indexes:
        image = io.imread(image_list[i])
        resize_image, t = random_aspect_ratio_and_resize_image(image)
        images.append(resize_image)
        ratios.append(t)
    X = np.stack(images, axis=0)
    T.append(ratios)

    return cuda.to_gpu(X), cuda.to_gpu(T)


if __name__ == '__main__':
    # 超パラメータ
    max_iteration = 1000  # 繰り返し回数
    batch_size = 100  # ミニバッチサイズ
    valid_size = 20000

    image_list = []
    images = []
    ratios = []
    T = []

    f = open(r"file_list.txt", "r")
    for path in f:
        path = path.strip()
        image_list.append(path)
    f.close()

    train_image_list = image_list[:-valid_size]
    valid_image_list = image_list[-valid_size:]

    num_train = len(train_image_list)
    num_batches = num_train / batch_size

    time_origin = time.time()

    for epoch in range(max_iteration):
        time_begin = time.time()
        permu = range(num_train)
        for indexes in tqdm.tqdm(np.array_split(permu, num_batches)):
            for i in indexes:
                image = io.imread(image_list[i])
                h_image, w_image = image.shape[:2]

                while True:
                    aspect_ratio = np.random.rand() * 2  # 0.5~2の乱数を生成
                    if aspect_ratio > 0.5:
                        break

                if np.random.rand() > 0.5:  # 半々の確率で
                    w_image = w_image * aspect_ratio

                square_image = transform.resize(image, (500, 500))

                if h_image >= w_image:
                    h_image = int(h_image * (500.0 / w_image))
                    if (h_image % 2) == 1:
                        h_image = h_image + 1
                    w_image = 500
                    resize_image = transform.resize(image, (h_image, w_image))
                    diff = h_image - w_image
                    margin = int(diff / 2)
                    if margin == 0:
                        square_image = resize_image
                    else:
                        square_image = resize_image[margin:-margin, :]
                else:
                    w_image = int(w_image * (500.0 / h_image))
                    if (w_image % 2) == 1:
                        w_image = w_image + 1
                    h_image = 500
                    resize_image = transform.resize(image, (h_image, w_image))
                    diff = w_image - h_image
                    margin = int(diff / 2)
                    if margin == 0:
                        square_image = resize_image
                    else:
                        square_image = resize_image[:, margin:-margin]

                images.append(square_image)
                ratios.append(aspect_ratio)
            X = np.stack(images, axis=0)
            T.append(ratios)

        time_end = time.time()
        epoch_time = time_end - time_begin
        total_time = time_end - time_origin

        # 訓練データでの結果を表示
        print "epoch:", epoch
        print "time", epoch_time, "(", total_time, ")"
