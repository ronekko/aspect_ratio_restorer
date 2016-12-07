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


def generate_image(model, X, T, max_iteration, a):
    print T[0], np.exp(T[0])
    plt.imshow(X[0][0]/256.0)
    plt.show()
    X_data = Variable(cuda.to_gpu(X))
    for epoch in range(max_iteration):
        y = model.forward(X_data, True)
        y.grad = cuda.cupy.ones(y.data.shape, dtype=np.float32)
        y.backward(retain_grad=True)
        X_data = Variable(X_data.data + a * X_data.grad)
        X_new = cuda.to_cpu(X_data.data)
        X_new = np.transpose(X_new, (0, 2, 3, 1))
    print y.data[0], cuda.cupy.exp(y.data[0])
    plt.imshow(X_new[0]/256.0)
    plt.show()

if __name__ == '__main__':
    # 超パラメータ
    max_iteration = 500  # 繰り返し回数
    batch_size = 100
    num_train = 20000
    num_test = 100
    output_size = 256
    crop_size = 224
    aspect_ratio_max = 3
    aspect_ratio_min = 1.0
    a = 100
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
            r = 1/2.0
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
    X_data = Variable(cuda.to_gpu(X))
    for epoch in range(max_iteration):
        y = model.forward(X_data, True)
        y.grad = cuda.cupy.ones(y.data.shape, dtype=np.float32)
        y.backward(retain_grad=True)
        X_data = Variable(X_data.data + a * X_data.grad)
        X_new = cuda.to_cpu(X_data.data)
        X_new = np.transpose(X_new, (0,2,3,1))
        print y.data[0], cuda.cupy.exp(y.data[0])
        plt.imshow(X_new[0]/256.0)
        plt.show()

    y = model.forward(X_gpu, True)

    # 特徴マップを取得
    a = y.creator.inputs[0]
    l = []
    r = []
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    l5 = []
    temp = []
    while a.creator:
        if a.creator.label == 'Convolution2DFunction':
            l.append(cuda.to_cpu(a.data))
        if a.creator.label == 'ReLU':
            r.append(cuda.to_cpu(a.data))
        a = a.creator.inputs[0]

    # 特徴マップを表示
    for f in l[-1][1]:
        plt.matshow(f, cmap=plt.cm.gray)
        plt.show()

    l5 = []
    for c in range(68):
        temp = []
        for b in range(batch_size):
            temp.append(np.sum(r[1][b][c]))
        ave = np.average(temp)
        l5.append(ave)
    l4 = []
    for c in range(32):
        temp = []
        for b in range(batch_size):
            temp.append(np.sum(r[2][b][c]))
        ave = np.average(temp)
        l4.append(ave)
    l3 = []
    for c in range(32):
        temp = []
        for b in range(batch_size):
            temp.append(np.sum(r[3][b][c]))
        ave = np.average(temp)
        l3.append(ave)
    l2 = []
    for c in range(16):
        temp = []
        for b in range(batch_size):
            temp.append(np.sum(r[4][b][c]))
        ave = np.average(temp)
        l2.append(ave)
    l1 = []
    for c in range(16):
        temp = []
        for b in range(batch_size):
            temp.append(np.sum(r[5][b][c]))
        ave = np.average(temp)
        l1.append(ave)

    print 'layer1'
    plt.plot(l1)
    plt.show()
    print 'layer2'
    plt.plot(l2)
    plt.show()
    print 'layer3'
    plt.plot(l3)
    plt.show()
    print 'layer4'
    plt.plot(l4)
    plt.show()
    print 'layer5'
    plt.plot(l5)
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
