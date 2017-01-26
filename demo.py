# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:27:22 2017

@author: yamane
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import cv2

from chainer import serializers

import utility
import dog_data_regression_ave_pooling
import make_html


if __name__ == '__main__':
    save_root = r'C:\Users\yamane\Dropbox\correct_aspect_ratio\demo'
    txt_file = r'E:\voc2012\raw_dataset\output_size_256\output_size_256.txt'
    model_file = r'C:\Users\yamane\Dropbox\correct_aspect_ratio\dog_data_regression_ave_pooling\1485316413.27_asp_max_3.0\dog_data_regression_ave_pooling.npz'

    folder_name = model_file.split('\\')[-2]
    fix_folder = os.path.join(folder_name, 'fix')
    debased_folder = os.path.join(folder_name, 'debased')
    original_folder = os.path.join(folder_name, 'original')
    save_path_f = os.path.join(save_root, fix_folder)
    save_path_d = os.path.join(save_root, debased_folder)
    save_path_o = os.path.join(save_root, original_folder)
    if os.path.exists(save_path_f):
        pass
    elif os.path.exists(save_path_f):
        pass
    elif os.path.exists(save_path_f):
        pass
    else:
        os.makedirs(save_path_f)
        os.makedirs(save_path_d)
        os.makedirs(save_path_o)

    train_num = 16500
    test_num = 500
    asp_r_max = 2.0
    loss = []
    th = 0.11

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
        if np.random.rand() > 0.5:
            t_l = np.random.uniform(np.log(1.2), np.log(2.0))
        else:
            t_l = np.random.uniform(np.log(1/2.0), np.log(1/1.2))
        t_r = np.exp(t_l)
        # 画像読み込み
        img_ori = plt.imread(test_paths[i])
        img = utility.crop_center(img_ori)
        img = cv2.resize(img, (256, 256))
        dis_img = utility.change_aspect_ratio(img_ori, t_r)
        dis_img_ori = utility.change_aspect_ratio(img_ori, t_r)
        square_img = utility.crop_center(dis_img)
        resize_img = utility.crop_224(square_img)
#        resize_img = square_img
        x_bhwc = resize_img[None, ...]
        x_bchw = np.transpose(x_bhwc, (0, 3, 1, 2))
        x = x_bchw.astype(np.float32)
        y_l = model.predict(x, True)
        y_r = np.exp(y_l)
        fix_img = utility.change_aspect_ratio(dis_img_ori, 1/y_r)

        e_l = t_l - y_l[0][0]
        e_r = np.abs(t_r - y_r[0][0])
        loss.append(e_l)

#        file_name_f = os.path.join(save_path_f, ('%.18f' % e_l))
#        file_name_d = os.path.join(save_path_d, ('%.18f' % e_l))
#        file_name_o = os.path.join(save_path_o, ('%.18f' % e_l))

        print '[test_data]:', i+1
        print '[t_l]:', round(t_l, 4), '\t[t_r]:', round(t_r, 4)
        print '[y_l]:', round(y_l[0][0], 4), '\t[y_r]:', round(y_r[0][0], 4)
        print '[e_l]:', round(e_l, 4), '\t[e_r]:', round(e_r, 4)

#        plt.tick_params(labelbottom='off', labeltop='off', labelleft='off', labelright='off')
#        plt.tick_params(bottom='off', top='off', left='off', right='off')
#        plt.imshow(fix_img)
#        plt.savefig(file_name_f+'.png', format='png', bbox_inches='tight')
#
#        plt.tick_params(labelbottom='off', labeltop='off', labelleft='off', labelright='off')
#        plt.tick_params(bottom='off', top='off', left='off', right='off')
#        plt.imshow(dis_img_ori)
#        plt.savefig(file_name_d+'.png', format='png', bbox_inches='tight')
#
#        plt.tick_params(labelbottom='off', labeltop='off', labelleft='off', labelright='off')
#        plt.tick_params(bottom='off', top='off', left='off', right='off')
#        plt.imshow(img_ori)
#        plt.savefig(file_name_o+'.png', format='png', bbox_inches='tight')

        plt.figure(figsize=(16, 16))
        plt.subplot(131)
        plt.title('Distortion image')
        plt.tick_params(labelbottom='off', labeltop='off', labelleft='off', labelright='off')
        plt.tick_params(bottom='off', top='off', left='off', right='off')
        plt.imshow(dis_img_ori)
        plt.subplot(132)
        plt.title('Fixed image')
        plt.tick_params(labelbottom='off', labeltop='off', labelleft='off', labelright='off')
        plt.tick_params(bottom='off', top='off', left='off', right='off')
        plt.imshow(fix_img)
        plt.subplot(133)
        plt.title('Normal image')
        plt.tick_params(labelbottom='off', labeltop='off', labelleft='off', labelright='off')
        plt.tick_params(bottom='off', top='off', left='off', right='off')
        plt.imshow(img_ori)
        plt.show()

#    make_html.make_html(save_path_d)
#    make_html.make_html(save_path_f)
#    make_html.make_html(save_path_o)
    x = np.stack(loss, axis=0)
    mu, sigma = 100, 15

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.hist(x, bins=50)
    ax.set_title('first histogram $\mu=100,\ \sigma=15$')
    ax.set_xlabel('t-y')
    ax.set_ylabel('freq')
    fig.show()

    count = 0
    for i in range(500):
        if np.abs(x[i]) < th:
            count += 1
    print 'under', th, '=', count / 5.0, '%'
    print '[mean]:', np.mean(np.abs(loss))
    print '[median]:', np.median(np.abs(loss))
    print '[std]:', np.std(loss, ddof=1)
