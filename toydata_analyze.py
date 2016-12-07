# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 04:13:19 2016

@author: yamane
"""

import toydata
import toydata_regression
import numpy as np
import matplotlib.pyplot as plt
from chainer import cuda, serializers, Variable


def generate_image(model, X, T, max_iteration, a):
    print 'T:', T[0], 'exp(T):', np.exp(T[0])
    plt.matshow(X[0][0])
    plt.colorbar()
    plt.show()
    X_data = Variable(cuda.to_gpu(X))
    for epoch in range(max_iteration):
        y = model.forward(X_data, True)
        y.grad = cuda.cupy.ones(y.data.shape, dtype=np.float32)
        y.backward(retain_grad=True)
        X_data = Variable(X_data.data + a * X_data.grad)
        X_new = cuda.to_cpu(X_data.data)
        X_new = X_new.reshape(-1, 224, 224)
    print 'Y:', y.data[0], 'exp(Y):', cuda.cupy.exp(y.data[0])
    plt.matshow(X_new[0])
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    # 超パラメータ
    max_iteration = 500  # 繰り返し回数
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
    step_size = 0.01

    model = toydata_regression.Convnet().to_gpu()
    serializers.load_npz('model1480565177.66toydata2.npz', model)

    dataset = toydata.RandomCircleSquareDataset(
        image_size, circle_r_min, circle_r_max, size_min, size_max, p,
        output_size, aspect_ratio_max, aspect_ratio_min)

    X, T = dataset.minibatch_regression(1)

    # yを計算
    X_gpu = Variable(cuda.to_gpu(X))
    X_data = Variable(cuda.to_gpu(X))

    generate_image(model, X, T, max_iteration, step_size)

    y = model.forward(X_gpu, True)

    # 特徴マップを取得
    a = y.creator.inputs[0]
    l = []
    while a.creator:
        if a.creator.label == 'Convolution2DFunction':
            l.append(cuda.to_cpu(a.data))
        a = a.creator.inputs[0]

    # 特徴マップを表示
#    for f in l[-1][0]:
#        plt.matshow(f, cmap=plt.cm.gray)
#        plt.show()

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
