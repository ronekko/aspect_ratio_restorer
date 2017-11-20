# -*- coding: utf-8 -*-
"""
Created on Fri Jan 06 16:59:52 2017

@author: yamane
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import scipy.stats as st

import chainer
from chainer import serializers

import voc2012_regression_max_pooling
import load_datasets
import utility


def fix(model, stream, t, preprocess):
    for it in stream.get_epoch_iterator():
        x, t = load_datasets.data_crop(it[0], random=False, t=t,
                                       preprocess=preprocess)
    y = model.predict(x, True)
    error = t - y
    error_abs = np.abs(t - y)
    return error, error_abs


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    # https://matplotlib.org/examples/pylab_examples/histogram_percent_demo.html
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


def draw_graph(loss, success_asp, num_test, t_list, save_path,
               confidence_level=0.95):
    hist_file = os.path.join(save_path, 'error_hist')
    average_abs_file = os.path.join(save_path, 'average_abs')
    loss_file = os.path.join(save_path, 'loss.npy')
    dot_file = os.path.join(save_path, 'dot_hist')
    average_asp_file = os.path.join(save_path, 'average_asp')
    average_asp_abs_file = os.path.join(save_path, 'average_asp_abs')

    loss_abs = np.abs(loss)
    num_t, num_images = loss_abs.shape[:2]
    prot_t = []
    for i in t_list:
        prot_t.append(round(i, 1))
    threshold = np.log(success_asp)
    base_line = np.full((num_test,), threshold)

    # histogram of signed errors
    plt.rcParams["font.size"] = 14
    plt.figure(figsize=(10, 3))

    plt.hist(loss.ravel(), density=True, bins=101, range=(-1.0, 1.0),
             histtype='stepfilled')
    plt.xlabel('Error', fontsize=20)
    plt.ylabel('Density', fontsize=20)

    plt.grid()
    plt.savefig(hist_file+'.png', format='png', bbox_inches='tight')
    plt.show()

    # per image, averaged over distortion levels (std version)
    plt.rcParams["font.size"] = 14
    mean_loss_abs = np.mean(loss_abs, axis=0)
    std_abs = np.std(loss_abs, axis=0)
    plt.figure(figsize=(10, 3))
    plt.errorbar(range(num_images), mean_loss_abs, marker='o', linewidth=0,
                 elinewidth=1.5, yerr=std_abs, label='avg. abs. error + std')
#    plt.plot(base_line, label='log(1.1303)')
    plt.plot(np.linspace(-1000, 1000, len(base_line)), base_line,
             label='avg. abs. human error')
    plt.legend(loc="upper left")
    plt.xlabel('Image ID of test data', fontsize=20)
    plt.ylabel('Avg abs. error in log', fontsize=20)
    plt.xlim(-1, 100)
    plt.ylim(0, max(mean_loss_abs)+max(std_abs)+0.1)
    plt.grid()
    plt.savefig(average_abs_file+'.png', format='png', bbox_inches='tight')
    plt.show()

    # per image, averaged over distortion levels (confidence interval version)
    plt.rcParams["font.size"] = 14
    mean_loss_abs = np.mean(loss_abs, axis=0)
    sems = st.sem(loss_abs, axis=0)
    lcbs, ucbs = st.t.interval(confidence_level, num_images - 1,
                               loc=mean_loss_abs, scale=sems)
    plt.figure(figsize=(10, 3))
    plt.errorbar(range(num_images), mean_loss_abs, marker='o', linewidth=0,
                 elinewidth=1.5, yerr=(lcbs, ucbs),
                 label='avg. abs. error + 95% CI')
    plt.plot(base_line, label='log(1.1303)')
    plt.legend(loc="upper left")
    plt.xlabel('Image ID of test data', fontsize=20)
    plt.ylabel('Avg abs. error in log', fontsize=20)
    plt.ylim(0, max(mean_loss_abs)+max(std_abs)+0.1)
    plt.grid()
    plt.savefig(average_abs_file+'.png', format='png', bbox_inches='tight')
    plt.show()

#    # per distortion, averaged signed errors (box plot)
#    plt.figure(figsize=(9, 3))
#    plt.boxplot(loss)
#    plt.xlim([np.log(1/3.5), np.log(3.5)])
#    plt.xticks(range(num_t), prot_t)
#    plt.title('Error for each aspect ratio in log scale', fontsize=24)
#    plt.xlabel('Order of aspect ratio in log scale', fontsize=24)
#    plt.ylabel('Error(t-y) in log scale', fontsize=24)
#    plt.grid()
#    plt.savefig(loss_file+'.png', format='png', bbox_inches='tight')
#    plt.show()

    # per distortion, averaged signed errors
    loss_dot = np.stack(loss, axis=0)
    loss_dot = loss_dot.reshape(num_t, num_test)
    average = np.mean(loss, axis=1)
    plt.figure(figsize=(9, 3))
    plt.plot(loss_dot, 'o', c='#348ABD')
    plt.plot(average, label='average')
    plt.xticks(range(num_t), prot_t)
    plt.title('Error for each aspect ratio in log scale', fontsize=24)
    plt.legend(loc="upper left", fontsize=20)
    plt.xlabel('Order of aspect ratio in log scale', fontsize=24)
    plt.ylabel('Error(t-y) in log scale', fontsize=24)
    plt.grid()
    plt.savefig(dot_file+'.png', format='png', bbox_inches='tight')
    plt.show()

    # per distortion, averaged signed errors
    plt.figure(figsize=(9, 3))
    plt.plot(average, label='average error')
    plt.plot(base_line, label='log(1.1303)')
    plt.plot(-base_line, label='log(1.1303^-1)')
    plt.xticks(range(num_t), prot_t)
    plt.xlim(0, num_t)
    plt.title('average Error for each aspect ratio in log scale', fontsize=24)
    plt.legend(loc="upper left", fontsize=20)
    plt.xlabel('Order of aspect ratio in log scale', fontsize=24)
    plt.ylabel('average Error(t-y) in log scale', fontsize=24)
    plt.grid()
    plt.savefig(average_asp_file+'.png', format='png', bbox_inches='tight')
    plt.show()

    # per distortion level, averaged over images (std version)
    average_abs = np.mean(loss_abs, axis=1)
    std_abs = np.std(loss_abs, axis=1)
    plt.figure(figsize=(10, 3))
    plt.errorbar(range(num_t), average_abs, yerr=std_abs,
                 label='avg. abs. error + std')
    plt.plot(np.linspace(-100, 100, len(base_line)), base_line,
                         label='avg. abs. human error')
    plt.xticks(range(num_t), prot_t)
    plt.xlim(-0.5, num_t-0.5)
    plt.ylim(0, max(mean_loss_abs)+0.05)
    plt.legend(loc="upper right")
    plt.xlabel('Distortion of aspect ratio in log scale', fontsize=20)
    plt.ylabel('Avg. abs. error in log', fontsize=20)
    plt.grid()
    plt.savefig(average_asp_abs_file+'.png', format='png', bbox_inches='tight')
    plt.show()

    # per distortion level, averaged over images (confidence interval version)
    average_abs = np.mean(loss_abs, axis=1)
    sems = st.sem(loss_abs, axis=1)
    lcbs, ucbs = st.t.interval(0.95, num_t - 1, loc=average_abs, scale=sems)
    plt.figure(figsize=(10, 3))
    plt.errorbar(range(num_t), average_abs, yerr=(lcbs, ucbs),
                 label='avg. abs. error + 95% CI')
    plt.plot(base_line, label='log(1.1303)')
    plt.xticks(range(num_t), prot_t)
    plt.xlim(0, num_t)
    plt.ylim(0, max(mean_loss_abs)+0.05)
    plt.legend(loc="upper right")
    plt.xlabel('Distortion of aspect ratio in log scale', fontsize=20)
    plt.ylabel('Avg. abs. error in log', fontsize=20)
    plt.grid()
    plt.savefig(average_asp_abs_file+'.png', format='jpg', bbox_inches='tight')
    plt.show()

    success = np.abs(loss) < threshold
    print('success rate of estimates (abs. error below thresh. {}):'.format(
        threshold))
    print('\ttotal average = {} %'.format(success.mean()))
    print('\timage-wise average = {} %'.format(success.mean(axis=0)))
    print('\tAR-wise average = {} %'.format(success.mean(axis=1)))
    print('model_file', model_file)


if __name__ == '__main__':
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    # テスト結果を保存するルートパス
    save_root = r'demo'
    # モデルのルートパス
    model_file = r'npz\dog_data_regression_ave_pooling.npz'
    batch_size = 100
    crop_size = 224  # 切り抜きサイズ
    num_train = 16500
    num_valid = 500
    num_test = 100
    success_asp = np.exp(0.12247601469)  # 修正成功とみなすアスペクト比
    num_split = 20  # 歪み画像のアスペクト比の段階
    # specify None or 'edge' or 'blur'
#    preprocesses = [None, 'edge', 'blur']
    preprocesses = [None]

    for preprocess in preprocesses:
        loss_list = []
        loss_abs_list = []
        t_list = []
        folder_name = model_file.split('\\')[-2]
        if preprocess:
            folder_name += '_' + preprocess

        num_t = num_split + 1
        t_step = np.log(3.0) * 2 / num_split
        t = np.log(1/3.0)
        for i in range(num_t):
            t_list.append(t)
            t = t + t_step

        # 結果を保存するフォルダを作成
        folder_path = utility.create_folder(save_root, folder_name)
        # モデル読み込み
        model = voc2012_regression_max_pooling.Convnet().to_gpu()
        # Optimizerの設定
        serializers.load_npz(model_file, model)
        # streamの取得
        streams = load_datasets.load_voc2012_stream(
            batch_size, num_train, num_valid, num_test)
        train_stream, valid_stream, test_stream = streams
        # アスペクト比ごとに歪み画像を作成し、修正誤差を計算
        for t in t_list:
#            print(t)
            with chainer.no_backprop_mode(), \
                    chainer.using_config('train', False):
                loss, loss_abs = fix(model, test_stream, t, preprocess)
            loss_list.append(loss)
            loss_abs_list.append(loss_abs)
        loss = np.array(loss_list)

        # 修正誤差をグラフに描画
        draw_graph(loss, success_asp, num_test, t_list,
                   folder_path)
