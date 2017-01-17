# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:12:33 2016

@author: yamane
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue

from chainer import cuda, serializers, Variable
import chainer.links as L

import dog_data_regression_ave_pooling
import load_datasets
import affine_pool


def generate_image(model, X, T, max_iteration, a):
    X_data = Variable(cuda.to_gpu(X))
    for epoch in range(max_iteration):
        print epoch
        y = model.forward(X_data, True)
        y.grad = cuda.cupy.ones(y.data.shape, dtype=np.float32)
        y.backward(retain_grad=True)
        X_data = Variable(cuda.cupy.clip((X_data.data + a * X_data.grad), 0, 1))
        X_new = cuda.to_cpu(X_data.data)
        X_new = X_new.reshape(-1, 224, 224)
    print 'origin_T:', T[0], 'exp(origin_T):', np.exp(T[0])
    print 'new_T:', y.data[0], 'exp(new_T):', cuda.cupy.exp(y.data[0])
    # 元のXを表示
#        print 'origin_T:', T[0], 'exp(origin_T):', np.exp(T[0])
    X = np.transpose(X, (0, 2, 3, 1))
    plt.imshow(X[0]/256.0, cmap=plt.cm.gray)
    plt.title("origin_X")
    plt.colorbar()
    plt.show()
    # 最適化後のXを表示
#        print 'new_T:', y.data[0], 'exp(new_T):', cuda.cupy.exp(y.data[0])
    X_new = np.transpose(X_new, (1, 2, 0))
    plt.imshow(X_new/256.0, cmap=plt.cm.gray)
    plt.title("new_X")
    plt.colorbar()
    plt.show()
    return X_new


def get_receptive_field(y):
    # 特徴マップを取得
    a = y.creator.inputs[0]
    l = []
    while a.creator:
        if a.creator.label == 'ReLU':
            l.append(cuda.to_cpu(a.data))
        a = a.creator.inputs[0]
    return l


def check_use_channel(l, layer):
    use_channel = []
    layer = len(l) - layer
    for c in range(l[layer].shape[1:2][0]):
        t = []
        for b in range(batch_size):
            t.append(np.sum(l[layer][b][c]))
        ave = np.average(t)
        use_channel.append(ave)
    return use_channel


if __name__ == '__main__':
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    # 超パラメータ
    batch_size = 500  # ミニバッチサイズ
    output_size = 256  # 生成画像サイズ
    crop_size = 224  # ネットワーク入力画像サイズ
    aspect_ratio_min = 1.0  # 最小アスペクト比の誤り
    aspect_ratio_max = 2.0  # 最大アスペクト比の誤り
    step_size = 0.01
    crop = True
    test = True
    r = 1.0
    channels = 64
    hdf5_filepath = r'E:\voc2012\raw_dataset\output_size_500\output_size_500.hdf5'  # データセットファイル保存場所
    model_file = r'C:\Users\yamane\Dropbox\correct_aspect_ratio\dog_data_regression_ave_pooling\1484311832.51_asp_max_3.0\dog_data_regression_ave_pooling.npz'

    # stream作成
    dog_stream_train, dog_stream_test = load_datasets.load_dog_stream(
        hdf5_filepath, batch_size)
    # モデル読み込み
    model = dog_data_regression_ave_pooling.Convnet().to_gpu()
    # Optimizerの設定
    serializers.load_npz(model_file, model)

    fc = model.l1
    conv = L.Convolution2D(in_channels=channels, out_channels=1, ksize=1).to_gpu()
    conv.W.data[:] = fc.W.data.reshape(1, 64, 1, 1)
    conv.b.data[:] = fc.b.data

    queue = Queue(10)
    process = Process(target=load_datasets.load_data,
                      args=(queue, dog_stream_test, crop,
                            aspect_ratio_max, aspect_ratio_min,
                            output_size, crop_size, test, r))
    process.start()

    X_test, T_test = queue.get()
    X_test, T_test = X_test[126:127], T_test[126:127]
    process.terminate()
    X_test_gpu = Variable(cuda.to_gpu(X_test))
    # yを計算
    y_test = model.forward(X_test_gpu, True)
#    y_test = affine_pool.affine_pool(model, conv, X_test_gpu, True)

    print 'r', r
    print 'y', cuda.cupy.exp(y_test.data[0])

    # 特徴マップを取得
    l = get_receptive_field(y_test)
    # 特徴マップを表示
    for f in l[0][0]:
        plt.matshow(f, cmap=plt.cm.gray)
        plt.show()

#    # 特徴マップの使用率を取得
#    l5 = check_use_channel(l, 5)
#    l4 = check_use_channel(l, 4)
#    l3 = check_use_channel(l, 3)
#    l2 = check_use_channel(l, 2)
#    l1 = check_use_channel(l, 1)
#    # 特徴マップの使用率を表示
#    plt.plot(l1)
#    plt.title("layer1")
#    plt.show()
#    plt.plot(l2)
#    plt.title("layer2")
#    plt.show()
#    plt.plot(l3)
#    plt.title("layer3")
#    plt.show()
#    plt.plot(l4)
#    plt.title("layer4")
#    plt.show()
#    plt.plot(l5)
#    plt.title("layer5")
#    plt.show()
#    # 出力に対する入力の勾配を可視化
#    y_test.grad = cuda.cupy.ones(y_test.data.shape, dtype=np.float32)
#    y_test.backward(retain_grad=True)
#    grad = X_test_gpu.grad
#    grad = cuda.to_cpu(grad)
#    for c in grad[0]:
#        plt.imshow(c/256.0, vmin=0.035, vmax=-0.035, cmap=plt.cm.bwr)
#        plt.colorbar()
#        plt.show()

    # 入力画像を表示
    for c in X_test:
        c = np.transpose(c, (1, 2, 0))
        plt.imshow(c/256.0, cmap=plt.cm.gray)
        plt.colorbar()
        plt.show()

    print 'model_file', model_file
