# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 04:13:19 2016

@author: yamane
"""

import toydata
import toydata_regression
import numpy as np
import time
import tqdm
import h5py
import copy
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
from chainer import cuda, optimizers, Chain, serializers,Variable
import chainer.functions as F
import chainer.links as L
import cv2
from dog_data_regression import Convnet


if __name__ == '__main__':
    # 超パラメータ
    max_iteration = 100  # 繰り返し回数
    batch_size = 100  # ミニバッチサイズ
    num_train = 5000
    num_valid = 100
    learning_rate = 0.001
    image_size = 500
    circle_r_min = 50
    circle_r_max = 150
    size_min = 50
    size_max = 200
    p = [0.3, 0.3, 0.4]
    output_size = 224
    aspect_ratio_max = 4
    aspect_ratio_min = 2

    model = toydata_regression.Convnet().to_gpu()
    serializers.load_npz('model1480549145.49toydata2.npz',model)

    dataset = toydata.RandomCircleSquareDataset(
        image_size, circle_r_min, circle_r_max, size_min, size_max, p,
        output_size, aspect_ratio_max, aspect_ratio_min)

    X, T = dataset.minibatch_regression(1)

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
    for c in grad[0]:
        plt.matshow(c, cmap=plt.cm.bwr)
        plt.colorbar()
        plt.show()
    for c in X[0]:
        plt.matshow(c, cmap=plt.cm.gray)
        plt.colorbar()
        plt.show()
