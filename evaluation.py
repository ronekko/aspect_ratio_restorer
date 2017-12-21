# -*- coding: utf-8 -*-
"""
Created on Fri Jan 06 16:59:52 2017

@author: yamane
"""

import os
from pathluib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

import chainer
from chainer import serializers

import main_resnet
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


class TransformTestdata(object):
    """
    An input must be a CHW shaped, RGB ordered, [0, 255] valued image.
    """
    def __init__(self, max_horizontal_factor=4.0,
                 scaled_size=256, crop_size=224,
                 p_blur=0.1, blur_max_ksize=5,
                 p_add_lines=0.1,  max_num_lines=2):
        if not 0 <= p_blur <= 1:
            raise ValueError('p_blur must be "0 <= p_blur <=1".')
        if not 0 <= p_add_lines <= 1:
            raise ValueError('p_add_lines must be "0 <= p_add_lines <=1".')
        self.max_horizontal_factor = max_horizontal_factor
        self.scaled_size = scaled_size
        self.crop_size = crop_size
        self.p_blur = p_blur
        self.blur_max_ksize = blur_max_ksize
        self.p_add_lines = p_add_lines
        self.max_num_lines = max_num_lines

    def __call__(self, chw):
        chw = chw.astype(np.uint8)
        if self.p_blur > 0:
            chw = random_blur(chw, self.p_blur, self.blur_max_ksize)
        if self.p_add_lines > 0:
            chw = add_random_lines(chw, self.p_add_lines, self.max_num_lines)
        chw = transforms.random_flip(chw, False, True)
        chw, param_stretch = random_stretch(chw, self.max_horizontal_factor,
                                            return_param=True)
        chw = inscribed_center_crop(chw)
        chw = transforms.scale(chw, self.scaled_size)
#        chw = transforms.center_crop(chw, (256, 256))
        chw = transforms.random_crop(chw, (self.crop_size, self.crop_size))
        chw = chw.astype(np.float32) / 256.0

        return chw, param_stretch['log_ar'].astype(np.float32)

def load_test_dataset(filepath, batch_size, scaled_size, crop_size):
    dataset = chainer.datasets.ImageDataset(filepath)

    _, test_raw = chainer.datasets.split_dataset(dataset, 17000)
    test_raw, _ = chainer.datasets.split_dataset(test_raw, 100)

    transform = Transform(
        max_horizontal_factor, scaled_size, crop_size, p_blur, blur_max_ksize)
    test = chainer.datasets.TransformDataset(test_raw, transform)

    it_train = MultiprocessIterator(train, batch_size, True, shuffle_train, 5)
    it_valid = MultiprocessIterator(valid, batch_size, True, False, 1, 5)
    it_test = MultiprocessIterator(test, batch_size, True, False, 1, 1)
    return it_train, it_valid, it_test


if __name__ == '__main__':
    save_root = r'evaluation_results'
    # モデルのルートパス
    model_file = '0.000312133168336004, 20171206T000238, 4c22664.chainer'
    data_file_path = 'E:/voc2012/rgb_jpg_paths_for_paper_v1.3.txt'
    batch_size = 100
    crop_size = 224  # 切り抜きサイズ
    num_train = 16500
    num_valid = 500
    num_test = 100
    success_asp = np.exp(0.12247601469)  # 修正成功とみなすアスペクト比
    ar_interval = (-3, 3)
    num_split = 21  # 歪み画像のアスペクト比の段階
    # specify None or 'edge' or 'blur'
#    preprocesses = [None, 'edge', 'blur']
    preprocesses = [None]

    # モデル読み込み
    model_path = Path(model_file)
    model = main_resnet.Resnet()
    serializers.load_npz(model_file, model)
    model.to_gpu()

    for preprocess in preprocesses:
        loss_list = []
        loss_abs_list = []
        output_folder_name = model_path.stem
        if preprocess:
            output_folder_name = output_folder_name / ', ' + preprocess

        # 結果を保存するフォルダを作成
        output_folder_path = Path(save_root) / output_folder_name
        output_folder_path.mkdir(parents=True)

        t_list = np.linspace(
            -np.log(ar_interval), np.log(ar_interval), num_split)

        # streamの取得
        streams = load_datasets.load_voc2012_stream(
            batch_size, num_train, num_valid, num_test)
        train_stream, valid_stream, test_stream = streams
        # アスペクト比ごとに歪み画像を作成し、修正誤差を計算
        for t in t_list:
            with chainer.no_backprop_mode(), \
                    chainer.using_config('train', False):
                loss, loss_abs = fix(model, test_stream, t, preprocess)
            loss_list.append(loss)
            loss_abs_list.append(loss_abs)
        loss = np.array(loss_list)

        # 修正誤差をグラフに描画
        draw_graph(loss, success_asp, num_test, t_list,
                   folder_path)
