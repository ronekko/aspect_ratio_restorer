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
    txt_file = r'E:\voc\variable_dataset\output_size_256\output_size_256.txt'
    model_file = r'C:\Users\yamane\Dropbox\correct_aspect_ratio\dog_data_regression_ave_pooling\1485768519.06_asp_max_4.0\dog_data_regression_ave_pooling.npz'

    crop_size = 224
    num_train = 17000
    num_test = 100
    th = np.log(1.1)
    num_split = 50
    test = True

    loss = []
    loss_abs = []
    t_list = []
    base_line = np.ones((num_test,))

    num_t = num_split + 1
    t_step = np.log(3.0) * 2 / num_split
    t = np.log(1/3.0)
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
            dis_img = utility.change_aspect_ratio(img, t_r)
            square_img = utility.crop_center(dis_img)
            resize_img = cv2.resize(square_img, (256, 256))
            crop_img = utility.crop_224(resize_img)
            x_list.append(crop_img)

        x_bhwc = np.stack(x_list, axis=0)
        x_bchw = np.transpose(x_bhwc, (0, 3, 1, 2))
        x_bchw = x_bchw.astype(np.float32)
        y_l = model.predict(x_bchw, test)
        y_r = np.exp(y_l)

        e_l = t_l - y_l
        e_l_abs = np.abs(t_l - y_l)
        e_r = t_r - y_r
        loss.append(e_l)
        loss_abs.append(e_l_abs)

    for i in range(100):
        base_line[i] = th

    error = np.stack(loss, axis=0)
    error_flat = error.flat
    a = error_flat
#    if np.abs(max(a)) > np.abs(min(a)):
#        max_value = np.abs(max(a))
#    else:
#        max_value = np.abs(min(a))

    mean_value_abs = np.mean(loss_abs, axis=0)
    plt.figure(figsize=(16, 12))
    plt.plot(mean_value_abs)
    plt.plot(base_line, 'r-')
    plt.title('average absolute Error for each test data', fontsize=28)
    plt.legend(["average Error", "Error=0.0953"], loc="upper left")
    plt.xlabel('Order of test data number', fontsize=28)
    plt.ylabel('average Error(|t-y|)', fontsize=28)
    plt.ylim(0, max(mean_value_abs)+0.01)
    plt.grid()
    plt.show()

    losses = np.stack(loss, axis=0)
    losses = losses.reshape(num_t, num_test)
    average = np.mean(loss, axis=1)
    plt.figure(figsize=(16, 12))
    plt.plot(losses, 'o', c='#348ABD')
    plt.plot(average, 'r-', label='average Error')
#    plt.xlim([np.log(1/3.5), np.log(3.5)])
    plt.xticks(range(num_t), t_list)
    plt.title('Order of r in log scale vs. Error(t-y)', fontsize=28)
    plt.legend(loc="upper right")
    plt.xlabel('Order of r in log scale', fontsize=28)
    plt.ylabel('Error(t-y)', fontsize=28)
    plt.grid()
    plt.show()

    plt.figure(figsize=(16, 12))
    plt.boxplot(loss)
    plt.xlim([np.log(1/3.5), np.log(3.5)])
    plt.xticks(range(num_t), t_list)
    plt.title('Order of r in log scale vs. Error(t-y)', fontsize=28)
    plt.xlabel('Order of r in log scale', fontsize=28)
    plt.ylabel('Error(t-y)', fontsize=28)
    plt.grid()
    plt.show()

    count = 0
    for i in range(100):
        if mean_value_abs[i] < th:
            count += 1
    print 'under', th, '=', count, '%'
    print 'num_test', num_test
    print 'model_file', model_file
