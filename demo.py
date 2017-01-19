# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:27:22 2017

@author: yamane
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import cv2

from chainer import serializers

import utility
import dog_data_regression_ave_pooling


if __name__ == '__main__':
    save_root = r'C:\Users\yamane\Dropbox\correct_aspect_ratio\demo'
    txt_file = r'E:\voc2012\raw_dataset\output_size_500\output_size_500.txt'
    model_file = r'C:\Users\yamane\Dropbox\correct_aspect_ratio\dog_data_regression_ave_pooling\1484660782.8_asp_max_3.0\dog_data_regression_ave_pooling.npz'

    folder_name = model_file.split('\\')[-2]
    save_path = os.path.join(save_root, folder_name)
    if os.path.exists(save_path):
        pass
    else:
        os.makedirs(save_path)

    train_num = 16500
    test_num = 500
    asp_r_max = 2.0

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
    test_paths = image_paths[train_num:train_num+test_num]

    for i in range(test_num):
        # アスペクト比を設定
        t_l = np.random.uniform(np.log(1/asp_r_max), np.log(asp_r_max))
        t_r = np.exp(t_l)
        # 画像読み込み
        img = plt.imread(test_paths[i])
        dis_img = utility.change_aspect_ratio(img, t_r)
        square_img = utility.crop_center(dis_img)
        resize_img = cv2.resize(square_img, (224, 224))
        x_bhwc = resize_img[None, ...]
        x_bchw = np.transpose(x_bhwc, (0, 3, 1, 2))
        x = x_bchw.astype(np.float32)
        y_l = model.predict(x, True)
        y_r = np.exp(y_l)
        fix_img = utility.change_aspect_ratio(dis_img, 1/y_r)

        e_l = np.abs(t_l - y_l[0][0])
        e_r = np.abs(t_r - y_r[0][0])

        file_name = os.path.join(save_path, str(e_r))

#        print '[test_data]:', i+1
#        print '[t_l]:', round(t_l, 4), '\t[t_r]:', round(t_r, 4)
#        print '[y_l]:', round(y_l[0][0], 4), '\t[y_r]:', round(y_r[0][0], 4)
#        print '[e_l]:', round(e_l, 4), '\t[e_r]:', round(e_r, 4)

        plt.tick_params(labelbottom='off', labeltop='off', labelleft='off', labelright='off')
        plt.tick_params(bottom='off', top='off', left='off', right='off')
        plt.imshow(fix_img)
        plt.savefig(file_name+'.png', format='png', bbox_inches='tight')

#        plt.figure(figsize=(16, 16))
#        plt.subplot(131)
#        plt.title('Distortion image')
#        plt.imshow(dis_img)
#        plt.subplot(132)
#        plt.title('Fixed image')
#        plt.imshow(fix_img)
#        plt.subplot(133)
#        plt.title('Normal image')
#        plt.imshow(img)
#        plt.show()
