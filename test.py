# -*- coding: utf-8 -*-
"""
Created on Fri Jan 06 16:59:52 2017

@author: yamane
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import cv2

from chainer import serializers

import dog_data_regression_ave_pooling
import utility


if __name__ == '__main__':
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    # 超パラメータ
    save_root = r'C:\Users\yamane\Dropbox\correct_aspect_ratio\demo'
    txt_file = r'E:\voc2012\raw_dataset\output_size_256\output_size_256.txt'
    model_file = r'C:\Users\yamane\Dropbox\correct_aspect_ratio\dog_data_regression_ave_pooling\1485316413.27_asp_max_3.0\\dog_data_regression_ave_pooling.npz'

    crop_size = 224
    num_train = 16500
    num_test = 500
    th = 0.1
    num_split = 50
    test = True

    loss = []
    loss_abs = []
    t_list = []

    num_t = num_split + 1
    t_step = np.log(2.0) * 2 / num_split
    t = np.log(0.5)
    for i in range(num_t):
        t_list.append(t)
        t = t + t_step

    # モデル読み込み
    model = dog_data_regression_ave_pooling.Convnet().to_gpu()
    # Optimizerの設定
    serializers.load_npz(model_file, model)

    # テキストファイル読み込み
    image_paths = []
    f = open(txt_file, 'r')
    for path in f:
        path = path.strip()
        image_paths.append(path)
    test_paths = image_paths[num_train:num_train+num_test]

    for t in t_list:
        print t
        # アスペクト比を設定
        t_l = t
        t_r = np.exp(t_l)
        x_list = []

        for i in range(num_test):
            # 画像読み込み
            img = plt.imread(test_paths[i])
#            img = utility.crop_center(img)
#            img = cv2.resize(img, (256, 256))
            dis_img = utility.change_aspect_ratio(img, t_r)
            square_img = utility.crop_center(dis_img)
#            square_img = cv2.resize(square_img, (256, 256))
            crop_img = utility.crop_224(square_img)
#            resize_img = square_img
            crop_img = crop_img.astype(np.float32)
            x_list.append(crop_img)

        x_bhwc = np.stack(x_list, axis=0)
        x_bchw = np.transpose(x_bhwc, (0, 3, 1, 2))
        y_l = model.predict(x_bchw, test)
        y_r = np.exp(y_l)

        e_l = y_l - t_l
        e_l_abs = np.abs(y_l - t_l)
        e_r = y_r - t_r
        loss.append(e_l)
        loss_abs.append(e_l_abs)

    mean_value = np.mean(loss_abs, axis=0)
    plt.figure(figsize=(16, 12))
    plt.plot(mean_value)
    plt.title('average error for each test data')
    plt.xlabel('Order of test data number')
    plt.ylabel('average error of prediction in log scale')
    plt.ylim(0, max(mean_value)+0.01)
    plt.grid()
    plt.show()

    ee = np.stack(loss, axis=0)
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

    count = 0
    for i in range(500):
        if mean_value[i] < th:
            count += 1
    print 'under', th, '=', count / 5.0, '%'
    print 'num_test', num_test
    print 'model_file', model_file
