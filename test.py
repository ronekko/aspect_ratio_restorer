# -*- coding: utf-8 -*-
"""
Created on Fri Jan 06 16:59:52 2017

@author: yamane
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue

from chainer import serializers

import dog_data_regression
import dog_data_regression_ave_pooling
import load_datasets


if __name__ == '__main__':
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    # 超パラメータ
    max_iteration = 150  # 繰り返し回数
    batch_size = 500  # ミニバッチサイズ
    num_train = 20000  # 学習データ数
    num_test = 500  # 検証データ数
    crop = True
    test = True
    hdf5_filepath = r'E:\voc2012\raw_dataset\output_size_500\output_size_500.hdf5'  # データセットファイル保存場所
    model_file = r'C:\Users\yamane\Dropbox\correct_aspect_ratio\dog_data_regression_ave_pooling\1484830422.72_asp_max_3.0\dog_data_regression_ave_pooling.npz'
    aspect_ratio_max = 2.0
    aspect_ratio_min = 1.0
    output_size = 256
    crop_size = 224

    t_losses = []
    t_list = []
    num_split = 50
    num_t = num_split + 1
    t_step = np.log(2.0) * 2 / num_split
    t = np.log(0.5)
    for i in range(num_t):
        t_list.append(t)
        t = t + t_step

    # バッチサイズ計算
    num_batches_test = num_test / batch_size
    # stream作成
    dog_stream_train, dog_stream_test = load_datasets.load_dog_stream(
        hdf5_filepath, batch_size)
    # モデル読み込み
    model = dog_data_regression_ave_pooling.Convnet().to_gpu()
    # Optimizerの設定
    serializers.load_npz(model_file, model)

    for r in np.exp(t_list):
        queue = Queue(10)
        process = Process(target=load_datasets.load_data,
                          args=(queue, dog_stream_test, crop, aspect_ratio_max,
                                aspect_ratio_min, output_size, crop_size,
                                test, r))
        process.start()
        t_loss = []
        loss = []
        for i in range(num_batches_test):
            # テスト用のデータを取得
            X_test, T_test = queue.get()
            # 復元結果を表示
            t_loss = dog_data_regression.test_output(model, X_test, T_test,
                                                     t_loss)
        process.terminate()
        t_losses.append(t_loss[0])
    sum_value = np.ndarray((500,))
    for a in range(500):
        for b in range(51):
            sum_value[a] += np.abs(t_losses[b][a])
    mean_value = sum_value / 500.0
    plt.figure(figsize=(16, 12))
    plt.plot(mean_value)
    plt.title('average error for each test data')
    plt.xlabel('Order of test data number')
    plt.ylabel('average error of prediction in log scale')
    plt.ylim(0, max(mean_value)+0.01)
    plt.grid()
    plt.show()

    e = t_losses
    ee = np.ndarray((num_t, num_test, 1))
    for i in range(len(ee)):
        ee[i] = e[i]
    ee = ee.reshape(num_t, num_test)
    eee = np.ndarray((num_t, 1))
    for i in range(len(ee)):
        eee[i] = np.mean(ee[i])
    eee = eee.reshape(num_t)
    plt.figure(figsize=(16, 12))
    plt.plot(ee, 'o', c='#348ABD')
    plt.plot(eee, 'r-')
    plt.xlim([np.log(1/2.5), np.log(2.5)])
    plt.xticks(range(num_t), t_list)
    plt.title('DCT Coefficient Amplitude vs. Order of Coefficient')
    plt.xlabel('Order of AR in log scale')
    plt.ylabel('error of prediction in log scale')
    plt.grid()
    plt.show()

    print 'num_test', num_test
    print 'model_file', model_file
