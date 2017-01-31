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
    save_root = r'E:\demo'
    txt_file = r'E:\voc\variable_dataset\output_size_256\output_size_256.txt'
    model_file = r'C:\Users\yamane\Dropbox\correct_aspect_ratio\dog_data_regression_ave_pooling\1485686792.4_asp_max_4.0\dog_data_regression_ave_pooling.npz'

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

    num_train = 17000
    num_test = 100
    asp_r_max = 3.0
    th = np.log(1.1)
    base_line = np.ones((num_test,))

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

    loss = []
    loss_abs = []
    for i in range(num_test):
        # アスペクト比を設
        t_l = np.random.uniform(np.log(1/asp_r_max), np.log(asp_r_max))
        t_r = np.exp(t_l)
        # 画像読み込み
        img = plt.imread(test_paths[i])
        dis_img = utility.change_aspect_ratio(img, t_r)
        square_img = utility.crop_center(dis_img)
        resize_img = cv2.resize(square_img, (256, 256))
        crop_img = utility.crop_224(resize_img)
        x_bhwc = crop_img[None, ...]
        x_bchw = np.transpose(x_bhwc, (0, 3, 1, 2))
        x = x_bchw.astype(np.float32)
        y_l = model.predict(x, True)
        y_r = np.exp(y_l)
        fix_img = utility.change_aspect_ratio(dis_img, 1/y_r)

        e_l = t_l - y_l
        e_l_abs = np.abs(t_l - y_l)
        e_r = t_r - y_r
        loss.append(e_l)
        loss_abs.append(e_l_abs)

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
#        plt.imshow(dis_img)
#        plt.savefig(file_name_d+'.png', format='png', bbox_inches='tight')
#
#        plt.tick_params(labelbottom='off', labeltop='off', labelleft='off', labelright='off')
#        plt.tick_params(bottom='off', top='off', left='off', right='off')
#        plt.imshow(img)
#        plt.savefig(file_name_o+'.png', format='png', bbox_inches='tight')

#        plt.figure(figsize=(16, 16))
#        plt.subplot(131)
#        plt.title('Distortion image')
#        plt.tick_params(labelbottom='off', labeltop='off', labelleft='off', labelright='off')
#        plt.tick_params(bottom='off', top='off', left='off', right='off')
#        plt.imshow(dis_img)
#        plt.subplot(132)
#        plt.title('Fixed image')
#        plt.tick_params(labelbottom='off', labeltop='off', labelleft='off', labelright='off')
#        plt.tick_params(bottom='off', top='off', left='off', right='off')
#        plt.imshow(fix_img)
#        plt.subplot(133)
#        plt.title('Normal image')
#        plt.tick_params(labelbottom='off', labeltop='off', labelleft='off', labelright='off')
#        plt.tick_params(bottom='off', top='off', left='off', right='off')
#        plt.imshow(img)
#        plt.show()

#    make_html.make_html(save_path_d)
#    make_html.make_html(save_path_f)
#    make_html.make_html(save_path_o)

    for i in range(num_test):
        base_line[i] = th

    error_abs = np.stack(loss_abs, axis=0)
    error_abs = error_abs.reshape(num_test, 1)
    error = np.stack(loss, axis=0)
    error = error.reshape(num_test, 1)

    if np.abs(max(error)) > np.abs(min(error)):
        max_value = np.abs(max(error))
    else:
        max_value = np.abs(min(error))

    plt.figure(figsize=(16, 12))
    plt.plot(error_abs)
    plt.plot(base_line, 'r-')
    plt.title('error abs for each test data')
    plt.legend(["error", "base line"], loc="upper right")
    plt.xlabel('Order of test data number')
    plt.ylabel('|t-y|')
    plt.ylim(0, max(error_abs)+0.01)
    plt.grid()
    plt.show()

    plt.figure(figsize=(16, 12))
    plt.plot(error)
    plt.plot(base_line, 'r-')
    plt.plot(-base_line, 'r-')
    plt.title('error for each test data')
    plt.legend(["error", "top base line", "bottom base line"], loc="upper right")
    plt.xlabel('Order of test data number')
    plt.ylabel('t-y')
    plt.ylim(-max_value-0.01, max_value+0.01)
    plt.grid()
    plt.show()

    mu, sigma = 100, 15
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(error, bins=25)
    ax.set_title('error histogram $\mu=100,\ \sigma=15$')
    ax.set_xlabel('t-y')
    ax.set_ylabel('freq')
    plt.xlim(-max_value-0.01, max_value+0.01)
    fig.show()

    count = 0
    for i in range(num_test):
        if loss_abs[i] < th:
            count += 1
    print 'under', th, '=', count, '%'
    print '[mean]:', np.mean(loss_abs)
