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
    batch_size = 100  # ミニバッチサイズ
    output_size = 256  # 生成画像サイズ
    crop_size = 224  # ネットワーク入力画像サイズ
    aspect_ratio_min = 1.0  # 最小アスペクト比の誤り
    aspect_ratio_max = 2.0  # 最大アスペクト比の誤り
    step_size = 0.01
    crop = True
    test = True
    r = [1.0, 1.0]
    images = []
    channels = 512
    hdf5_filepath = r'E:\voc\variable_dataset\output_size_256\output_size_256.hdf5'  # データセットファイル保存場所
    model_file = r'C:\Users\yamane\Dropbox\correct_aspect_ratio\dog_data_regression_ave_pooling\1485768519.06_asp_max_4.0\dog_data_regression_ave_pooling.npz'

    # モデル読み込み
    model = dog_data_regression_ave_pooling.Convnet().to_gpu()
    # Optimizerの設定
    serializers.load_npz(model_file, model)

    origin = plt.imread('test.jpg')

    for i in r:
        dis_img = utility.change_aspect_ratio(origin, i)
        square = utility.crop_center(dis_img)
        resize = cv2.resize(square, (256, 256))
        crop = utility.crop_224(resize)
        images.append(crop)
    x_bhwc = np.stack(images, axis=0)
    x_bchw = np.transpose(x_bhwc, (0, 3, 1, 2))
    x_bchw = x_bchw.astype(np.float32)
    X_test_gpu = Variable(cuda.to_gpu(x_bchw))

    # yを計算
    y_test = model.forward(X_test_gpu, True)
    # 特徴マップを取得
    l = get_receptive_field(y_test)
    # 特徴マップを表示
    f_yoko = l[-1][0]
    f_tate = l[-1][1]
    for i in range(64):
        plt.figure(figsize=(8, 8))
        plt.subplot(121)
        plt.title('yoko')
        plt.imshow(f_yoko[i], interpolation='nearest', cmap=plt.cm.gray)
        plt.colorbar()
        plt.subplot(122)
        plt.title('tate')
        plt.imshow(f_tate[i], interpolation='nearest', cmap=plt.cm.gray)
        plt.colorbar()
        plt.show()

    fc = model.l1
    conv = L.Convolution2D(in_channels=channels, out_channels=1, ksize=1).to_gpu()
    conv.W.data[:] = fc.W.data.reshape(1, 512, 1, 1)
    conv.b.data[:] = fc.b.data

    h_test, y_test = affine_pool.affine_pool(model, conv, X_test_gpu, True)
    h = cuda.to_cpu(h_test.data)
    if np.abs(np.min(h[0][0])) > np.abs(np.max(h[0][0])):
        max_value_yoko = np.abs(np.min(h[0][0]))
    else:
        max_value_yoko = np.abs(np.max(h[0][0]))

    if np.abs(np.min(h[1][0])) > np.abs(np.max(h[1][0])):
        max_value_tate = np.abs(np.min(h[1][0]))
    else:
        max_value_tate = np.abs(np.max(h[1][0]))

    plt.figure(figsize=(8, 8))
    plt.subplot(121)
    plt.title('yoko')
    plt.imshow(h[0][0], vmin=-max_value_yoko, vmax=max_value_yoko, interpolation='nearest', cmap=plt.cm.bwr)
    plt.colorbar()
    plt.subplot(122)
    plt.title('tate')
    plt.imshow(h[1][0], vmin=-max_value_tate, vmax=max_value_tate, interpolation='nearest', cmap=plt.cm.bwr)
    plt.colorbar()
    plt.show()

    # 入力画像を表示
    plt.figure(figsize=(8, 8))
    plt.subplot(121)
    plt.title('yoko')
    plt.imshow(x_bhwc[0])
    plt.subplot(122)
    plt.title('tate')
    plt.imshow(x_bhwc[1])
    plt.show()

    print 'r_yoko', r[0], 'r_tate', r[1]
    print 'y_yoko:', cuda.cupy.exp(y_test.data[0][0][0][0]), 'y_tate:', cuda.cupy.exp(y_test.data[1][0][0][0])
    print 'model_file', model_file
