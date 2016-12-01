# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:12:33 2016

@author: yamane
"""


import toydata
import numpy as np
import h5py
import matplotlib.pyplot as plt
from chainer import cuda, serializers, Variable
import cv2
from dog_data_regression import Convnet


if __name__ == '__main__':
    # 超パラメータ
    batch_size = 2
    num_train = 20000
    num_test = 100
    output_size = 256
    crop_size = 224
    aspect_ratio_max = 3
    aspect_ratio_min = 1.0
    file_path = r'E:\stanford_Dogs_Dataset\raw_dataset_binary\output_size_500\output_size_500.hdf5'
    model_name = 'model1480445096.37dog1.5.npz'
    test_data = range(num_train, num_train + num_test)
    model = Convnet().to_gpu()
    serializers.load_npz(model_name, model)

    dataset = h5py.File(file_path)
    image_features = dataset['image_features']
    r_min = aspect_ratio_min
    r_max = aspect_ratio_max

    num_batches = len(test_data) / batch_size

    for indexes in np.array_split(test_data, num_batches):
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
        T = np.array(ts, dtype=np.float32).reshape(-1, 1)

    # yを計算
    X_gpu = Variable(cuda.to_gpu(X))
    y = model.forward(X_gpu, True)

    # 特徴マップを取得
    a = y.creator.inputs[0]
    l = []
    while a.creator:
        if a.creator.label == 'Convolution2DFunction':
            l.append(cuda.to_cpu(a.data))
        a = a.creator.inputs[0]

    # 特徴マップを表示
    for f in l[-1][0]:
        plt.matshow(f, cmap=plt.cm.gray)
        plt.show()

    # 出力に対する入力の勾配を可視化
    y.grad = cuda.cupy.ones(y.data.shape, dtype=np.float32)
    y.backward(retain_grad=True)
    grad = X_gpu.grad
    grad = cuda.to_cpu(grad)
    for c in grad[1]:
        plt.matshow(c, cmap=plt.cm.bwr)
        plt.colorbar()
        plt.show()
    for c in X[1]:
        plt.matshow(c, cmap=plt.cm.gray)
        plt.colorbar()
        plt.show()
