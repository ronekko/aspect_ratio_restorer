# -*- coding: utf-8 -*-
"""
Created on Fri Jan 06 16:59:52 2017

@author: yamane
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from chainer import serializers

import dog_data_regression_ave_pooling
import load_datasets
import utility


def fix(model, stream, t):
    for it in stream_test.get_epoch_iterator():
        x, t = load_datasets.data_crop(it[0], random=False, t=t)
    y = model.predict(x, True)
    error = t - y
    error_abs = np.abs(t - y)
    return error, error_abs


def draw_graph(loss, loss_abs, success_asp, num_test, num_split, save_path):
    average_abs_file = os.path.join(save_path, 'average_abs')
    loss_file = os.path.join(save_path, 'loss')
    dot_file = os.path.join(save_path, 'dot_hist')
    average_asp_file = os.path.join(save_path, 'average_asp')
    average_asp_abs_file = os.path.join(save_path, 'average_asp_abs')
    num_t = num_split + 1
    threshold = np.log(success_asp)
    base_line = np.ones((num_test,))
    for i in range(num_test):
        base_line[i] = threshold

    mean_loss_abs = np.mean(loss_abs, axis=0)
    plt.figure(figsize=(18, 10))
    plt.plot(mean_loss_abs, label='average Error')
    plt.plot(base_line, label='log(1.1)')
    plt.title('average absolute Error for each test data', fontsize=28)
    plt.legend(loc="upper right")
    plt.xlabel('Order of test data number', fontsize=28)
    plt.ylabel('average Error(|t-y|) in log scale', fontsize=28)
    plt.ylim(0, max(mean_loss_abs)+0.01)
    plt.grid()
    plt.savefig(average_abs_file+'.jpg', format='jpg', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(18, 10))
    plt.boxplot(loss)
    plt.xlim([np.log(1/3.5), np.log(3.5)])
    plt.xticks(range(num_t), t_list)
    plt.title('Error for each aspect ratio in log scale', fontsize=28)
    plt.xlabel('Order of aspect ratio in log scale', fontsize=28)
    plt.ylabel('Error(t-y) in log scale', fontsize=28)
    plt.grid()
    plt.savefig(loss_file+'.jpg', format='jpg', bbox_inches='tight')
    plt.show()

    loss_dot = np.stack(loss, axis=0)
    loss_dot = loss_dot.reshape(num_t, num_test)
    average = np.mean(loss, axis=1)
    plt.figure(figsize=(18, 10))
    plt.plot(loss_dot, 'o', c='#348ABD')
    plt.plot(average, label='average')
    plt.xticks(range(num_t), t_list)
    plt.title('Error for each aspect ratio in log scale', fontsize=28)
    plt.legend(loc="upper right")
    plt.xlabel('Order of aspect ratio in log scale', fontsize=28)
    plt.ylabel('Error(t-y) in log scale', fontsize=28)
    plt.grid()
    plt.savefig(dot_file+'.jpg', format='jpg', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(18, 10))
    plt.plot(average, label='average Error')
    plt.plot(base_line, label='log(1.1)')
    plt.plot(-base_line, label='log(1.1^-1)')
    plt.xticks(range(num_t), t_list)
    plt.xlim(0, num_t)
    plt.title('average Error for each aspect ratio in log scale', fontsize=28)
    plt.legend(loc="upper right")
    plt.xlabel('Order of aspect ratio in log scale', fontsize=28)
    plt.ylabel('average Error(t-y) in log scale', fontsize=28)
    plt.grid()
    plt.savefig(average_asp_file+'.jpg', format='jpg', bbox_inches='tight')
    plt.show()

    average_abs = np.abs(average)
    plt.figure(figsize=(18, 10))
    plt.plot(average_abs, label='average absolute Error')
    plt.plot(base_line, label='log(1,1)')
    plt.xticks(range(num_t), t_list)
    plt.xlim(0, num_t)
    plt.title('average absolute Error for each aspect ratio in log scale',
              fontsize=28)
    plt.legend(loc="upper right")
    plt.xlabel('Order of aspect ratio in log scale', fontsize=28)
    plt.ylabel('Error(|t-y|)', fontsize=28)
    plt.grid()
    plt.savefig(average_asp_abs_file+'.jpg', format='jpg', bbox_inches='tight')
    plt.show()

    count = 0
    for i in range(num_test):
        if mean_loss_abs[i] < threshold:
            count += 1
    print 'under log(1.1) =', count, '%'
    print 'num_test', num_test
    print 'model_file', model_file


if __name__ == '__main__':
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    # テスト結果を保存するルートパス
    save_root = r'C:\Users\yamane\Dropbox\correct_aspect_ratio\demo'
    # hdf5ファイルのルートパス
    hdf5_filepath = r'E:\voc\variable_dataset\output_size_256\output_size_256.hdf5'
    # モデルのルートパス
    model_file = r'C:\Users\yamane\Dropbox\correct_aspect_ratio\dog_data_regression_ave_pooling\1485768519.06_asp_max_4.0\dog_data_regression_ave_pooling.npz'
    batch_size = 100
    crop_size = 224  # 切り抜きサイズ
    num_train = 16500
    num_valid = 500
    num_test = 100
    success_asp = 1.1  # 修正成功とみなすアスペクト比
    num_split = 50  # 歪み画像のアスペクト比の段階

    loss_list = []
    loss_abs_list = []
    t_list = []
    folder_name = model_file.split('\\')[-2]

    num_t = num_split + 1
    t_step = np.log(3.0) * 2 / num_split
    t = np.log(1/3.0)
    for i in range(num_t):
        t_list.append(t)
        t = t + t_step

    # 結果を保存するフォルダを作成
    folder_path = utility.create_folder(save_root, folder_name)
    # モデル読み込み
    model = dog_data_regression_ave_pooling.Convnet().to_gpu()
    # Optimizerの設定
    serializers.load_npz(model_file, model)
    # streamの取得
    stream_train, stream_valid, stream_test = load_datasets.load_dog_stream(
        hdf5_filepath, batch_size, num_train, num_valid, num_test)
    # アスペクト比ごとに歪み画像を作成し、修正誤差を計算
    for t in t_list:
        print t
        loss, loss_abs = fix(model, stream_test, t)
        loss_list.append(loss)
        loss_abs_list.append(loss_abs)
    # 修正誤差をグラフに描画
    draw_graph(loss_list, loss_abs_list, success_asp, num_test, num_split, folder_path)
