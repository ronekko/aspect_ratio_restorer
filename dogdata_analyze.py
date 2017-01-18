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
import utility
import cv2


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
    model_file = r'C:\Users\yamane\Dropbox\correct_aspect_ratio\dog_data_regression_ave_pooling\1484660782.8_asp_max_3.0\dog_data_regression_ave_pooling.npz'

    # モデル読み込み
    model = dog_data_regression_ave_pooling.Convnet().to_gpu()
    # Optimizerの設定
    serializers.load_npz(model_file, model)

    origin = plt.imread(r'dog_data_regression_ave_pooling\1484660782.8_asp_max_3.0\max\2009_003151.jpg')
    debased = utility.change_aspect_ratio(origin, r)
    square = utility.crop_center(debased)
    resize = cv2.resize(square, (224, 224))
    batch = resize[None, ...]
    batch = np.transpose(batch, (0, 3, 1, 2))
    X_test = batch.astype(np.float32)
    T_test = r
    X_test_gpu = Variable(cuda.to_gpu(X_test))

#    # yを計算
#    h_test, y_test = affine_pool.pool_affine(model, X_test_gpu, True)
#    # 特徴マップを取得
#    l = get_receptive_field(y_test)
#    # 特徴マップを表示
#    for f in l[0][0]:
#        plt.matshow(f, cmap=plt.cm.gray)
#        plt.colorbar()
#        plt.show()

    fc = model.l1
    conv = L.Convolution2D(in_channels=channels, out_channels=1, ksize=1).to_gpu()
    conv.W.data[:] = fc.W.data.reshape(1, 64, 1, 1)
    conv.b.data[:] = fc.b.data

    h_test, y_test = affine_pool.affine_pool(model, conv, X_test_gpu, True)
    h = cuda.to_cpu(h_test.data)
    plt.matshow(h[0][0]+np.absolute(np.min(h[0][0])), cmap=plt.cm.gray)
    plt.colorbar()
    plt.show()

    # 入力画像を表示
    for c in X_test:
        c = np.transpose(c, (1, 2, 0))
        plt.imshow(c/256.0, cmap=plt.cm.gray)
        plt.colorbar()
        plt.show()

    print 'r', r
    print 'y', cuda.cupy.exp(y_test.data[0])
    print 'model_file', model_file
