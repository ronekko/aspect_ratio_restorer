# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:27:22 2017

@author: yamane
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from chainer import serializers

import utility
import load_datasets
import dog_data_regression_ave_pooling
import make_html


def create_save_folder(save_root, model_file):
    folder_name = model_file.split('\\')[-2]
    fix_folder = os.path.join(folder_name, 'fix')
    distorted_folder = os.path.join(folder_name, 'distorted')
    original_folder = os.path.join(folder_name, 'original')
    save_path_f = os.path.join(save_root, fix_folder)
    save_path_d = os.path.join(save_root, distorted_folder)
    save_path_o = os.path.join(save_root, original_folder)
    if os.path.exists(save_path_f):
        pass
    elif os.path.exists(save_path_d):
        pass
    elif os.path.exists(save_path_o):
        pass
    else:
        os.makedirs(save_path_f)
        os.makedirs(save_path_d)
        os.makedirs(save_path_o)
    return save_path_f, save_path_d, save_path_o


def fix(model, stream, save_path_f, save_path_d, save_path_o):
    loss = []
    loss_abs = []
    for it in stream.get_epoch_iterator():
        x, t_l = load_datasets.data_crop(it[0])
        t_r = np.exp(t_l)
        y_l = model.predict(x, True)
        y_r = np.exp(y_l)
        e_l = t_l - y_l
        e_r = t_r - y_r
        e_l_abs = np.abs(t_l - y_l)
        for i in range(len(e_l)):
            file_name_f = os.path.join(save_path_f, ('%.18f' % e_l[i]))
            file_name_d = os.path.join(save_path_d, ('%.18f' % e_l[i]))
            file_name_o = os.path.join(save_path_o, ('%.18f' % e_l[i]))
            img = it[0][i]
            dis_img = dis_img = utility.change_aspect_ratio(img, t_r[i], 1)
            fix_img = utility.change_aspect_ratio(dis_img, 1/y_r[i], 1)

            print '[test_data]:', i+1
            print '[t_l]:', round(t_l[i], 4), '\t[t_r]:', round(t_r[i], 4)
            print '[y_l]:', round(y_l[i], 4), '\t[y_r]:', round(y_r[i], 4)
            print '[e_l]:', round(e_l[i], 4), '\t[e_r]:', round(e_r[i], 4)

            plt.figure(figsize=(16, 16))
#            plt.subplot(131)
#            plt.title('Distorted image')
            plt.tick_params(labelbottom='off', labeltop='off', labelleft='off',
                            labelright='off')
            plt.tick_params(bottom='off', top='off', left='off', right='off')
            plt.imshow(dis_img)
            plt.savefig(file_name_d+'.jpg', format='jpg', bbox_inches='tight')
#            plt.subplot(132)
#            plt.title('Fixed image')
            plt.tick_params(labelbottom='off', labeltop='off', labelleft='off',
                            labelright='off')
            plt.tick_params(bottom='off', top='off', left='off', right='off')
            plt.imshow(fix_img)
            plt.savefig(file_name_f+'.jpg', format='jpg', bbox_inches='tight')
#            plt.subplot(133)
#            plt.title('Normal image')
            plt.tick_params(labelbottom='off', labeltop='off', labelleft='off',
                            labelright='off')
            plt.tick_params(bottom='off', top='off', left='off', right='off')
            plt.imshow(img)
            plt.savefig(file_name_o+'.jpg', format='jpg', bbox_inches='tight')
#            plt.show()

    loss.append(e_l)
    loss_abs.append(e_l_abs)

    make_html.make_html(save_path_d)
    make_html.make_html(save_path_f)
    make_html.make_html(save_path_o)
    return loss, loss_abs


def draw_graph(loss, loss_abs, success_asp, num_test, save_root):
    loss_abs_file = os.path.join(save_root, 'loss_abs')
    loss_file = os.path.join(save_root, 'loss')
    loss_hist = os.path.join(save_root, 'loss_hist')
    threshold = np.log(success_asp)
    base_line = np.ones((num_test,))
    for i in range(num_test):
        base_line[i] = threshold

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
    plt.title('absolute Error for each test data', fontsize=28)
    plt.legend(["Error", "log(1.1)"], loc="upper right")
    plt.xlabel('Order of test data number', fontsize=28)
    plt.ylabel('Error(|t-y|) in log scale', fontsize=28)
    plt.ylim(0, max(error_abs)+0.01)
    plt.grid()
    plt.savefig(loss_abs_file+'.jpg', format='jpg', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(16, 12))
    plt.plot(error, label='Error')
    plt.plot(base_line, label="log(1.1)")
    plt.plot(-base_line, label="log(1.1^-1)")
    plt.title('Error for each test data', fontsize=28)
    plt.legend(loc="upper right")
    plt.xlabel('Order of test data number', fontsize=28)
    plt.ylabel('Error(t-y) in log scale', fontsize=28)
    plt.ylim(-max_value-0.01, max_value+0.01)
    plt.grid()
    plt.savefig(loss_file+'.jpg', format='jpg', bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(error, bins=25)
    ax.set_title('Error histogram', fontsize=28)
    ax.set_xlabel('Error(t-y) in log scale', fontsize=28)
    ax.set_ylabel('Percentage', fontsize=28)
    plt.xlim(-1, 1)
    plt.savefig(loss_hist+'.jpg', format='jpg', bbox_inches='tight')
    fig.show()

    count = 0
    for i in range(num_test):
        if loss_abs[0][i] < threshold:
            count += 1
    print 'under log(1.1) =', count, '%'
    print '[mean]:', np.mean(loss_abs)


if __name__ == '__main__':
    # テスト結果を保存する場所
    save_root = r'E:\demo'
    # テスト用のhdf5ファイルのルートパス
    hdf5_file = r'E:\voc\variable_dataset\output_size_256\output_size_256.hdf5'
    # テストに使うモデルのnpzファイルの場所
    model_file = r'C:\Users\yamane\Dropbox\correct_aspect_ratio\dog_data_regression_ave_pooling\1485768519.06_asp_max_4.0\dog_data_regression_ave_pooling.npz'
    num_train = 16500  # 学習データ数
    num_valid = 500  # 検証データ数
    num_test = 100  # テストデータ数
    asp_r_max = 3.0  # 歪み画像の最大アスペクト比
    success_asp = 1.1  # 修正成功とみなす修正画像のアスペクト比の最大値
    batch_size = 100

    # モデルのファイル名をフォルダ名にする
    folder_name = model_file.split('\\')[-2]

    # テスト結果を保存するフォルダを作成
    test_folder_path = utility.create_folder(save_root, folder_name)
    fix_folder_path = utility.create_folder(test_folder_path, 'fix')
    dis_folder_path = utility.create_folder(test_folder_path, 'distorted')
    ori_folder_path = utility.create_folder(test_folder_path, 'original')

    # モデル読み込み
    model = dog_data_regression_ave_pooling.Convnet().to_gpu()
    # Optimizerの設定
    serializers.load_npz(model_file, model)

    # streamを取得
    stream_train, stream_valid, stream_test = load_datasets.load_dog_stream(
        hdf5_file, batch_size)

    # 歪み画像の修正を実行
    loss, loss_abs = fix(model, stream_test,
                         fix_folder_path, dis_folder_path, ori_folder_path)

    # 修正結果の誤差を描画
    draw_graph(loss, loss_abs, success_asp, num_test, test_folder_path)
