# -*- coding: utf-8 -*-
"""
Created on Fri Jan 06 16:59:52 2017

@author: yamane
"""

import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

import chainer
from chainer.iterators import MultiprocessIterator
from chainer import serializers
from chainercv import transforms

import dataset_transform
import main_resnet
#import load_datasets


def draw_graph(l_pred, l_gt, baseline_ar, output_folder_path,
               confidence_level=0.95):
#    loss_file = os.path.join(save_path, 'error.npy')

    num_level, num_examples = l_pred.shape
    error = l_gt - l_pred
    error_abs = np.abs(error)
    threshold = np.log(baseline_ar)
    base_line = np.full((num_examples,), threshold)

    # histogram of signed errors
    plt.rcParams["font.size"] = 14
    plt.figure(figsize=(10, 3))
    plt.hist(error.ravel(), density=True, bins=101, range=(-1.0, 1.0),
             histtype='stepfilled')
    plt.xlabel('Error', fontsize=20)
    plt.ylabel('Density', fontsize=20)

    plt.grid()
    error_hist_path = str(output_folder_path / 'error_hist.png')
    plt.savefig(error_hist_path, format='png', bbox_inches='tight')
    plt.show()

    # per image, averaged over distortion levels (confidence interval version)
    plt.rcParams["font.size"] = 14
    mean_loss_abs = np.mean(error_abs, axis=0)
    sems = st.sem(error_abs, axis=0)
    lcbs, ucbs = st.t.interval(confidence_level, num_examples - 1,
                               loc=mean_loss_abs, scale=sems)
    plt.figure(figsize=(10, 3))
    plt.errorbar(range(num_examples), mean_loss_abs, marker='o', linewidth=0,
                 elinewidth=1.5, yerr=(lcbs, ucbs),
                 label='avg. abs. error + 95% CI')
    base_line = np.full((2,), threshold)
    xlim = [-1, 100]
    plt.plot(xlim, base_line, label='avg. abs. human error')
    plt.legend(loc="upper left")
    plt.xlabel('Image ID of test data', fontsize=20)
    plt.ylabel('Avg abs. error in log', fontsize=20)
    plt.xlim(*xlim)
    plt.ylim(0, threshold+0.01)
    plt.grid()
    image_wise_aae_ci_path = str(output_folder_path / 'image_wise_aae_ci.png')
    plt.savefig(image_wise_aae_ci_path, format='png', bbox_inches='tight')
    plt.show()

    # per distortion level, averaged over images (confidence interval version)
    average_abs = np.mean(error_abs, axis=1)
    sems = st.sem(error_abs, axis=1)
    lcbs, ucbs = st.t.interval(
        0.95, num_level - 1, loc=average_abs, scale=sems)
    plt.figure(figsize=(10, 3))
    plt.errorbar(l_gt[:, 0], average_abs, yerr=(lcbs, ucbs),
                 label='avg. abs. error + 95% CI')
    base_line = np.full((2,), threshold)
    unit = l_gt[1, 0] - l_gt[0, 0]
    xlim = l_gt[0, 0] - unit / 2, l_gt[-1, 0] + unit / 2
    plt.plot(xlim, base_line, label='avg. abs. human error')
    plt.xlim(*xlim)
    plt.ylim(0, threshold+0.01)
    plt.legend(loc="upper right")
    plt.xlabel('Distortion of aspect ratio in log scale', fontsize=20)
    plt.ylabel('Avg. abs. error in log', fontsize=20)
    plt.grid()
    level_wise_aae_ci_path = str(output_folder_path / 'level_wise_aae_ci.png')
    plt.savefig(level_wise_aae_ci_path, format='png', bbox_inches='tight')
    plt.show()

    # save text
    success = error_abs < threshold
    avg_abs_error = error_abs.mean()
    texts = []
    texts.append('average absolute error = {} ({} in AR)'.format(
        avg_abs_error, np.exp(avg_abs_error)))
    texts.append(
        'success rate of estimates (abs. error below thresh. {}):'.format(
            threshold))
    texts.append('total average = {} %'.format(success.mean()))
    texts.append('image-wise average = {} %'.format(success.mean(axis=0)))
    texts.append('AR-wise average = {} %'.format(success.mean(axis=1)))
    texts.append('model_file = {}'.format(model_file))
    text = '\n'.join(texts)
    print(text)
    accuracy_path = output_folder_path / 'accuracy.txt'
    with accuracy_path.open('w') as f:
        f.write(text)


class TransformTestdata(object):
    """
    An input must be a CHW shaped, RGB ordered, [0, 255] valued image.
    """
    def __init__(self, scaled_size, crop_size, log_ars, preprocess):
        self.scaled_size = scaled_size
        self.crop_size = crop_size
        self.log_ars = log_ars
        self.preprocess = preprocess

    def __call__(self, chw):
        chw = chw.astype(np.uint8)
#        if False:  # TODO: fix this
#            chw = random_blur(chw, self.p_blur, self.blur_max_ksize)
#        if False:  # TODO: fix this
#            chw = add_random_lines(chw, self.p_add_lines, self.max_num_lines)

        outputs = []
        chw_original = chw
        for log_ar in self.log_ars:
            chw = chw_original.copy()
            if self.preprocess == 'edge':
                chw = edge(chw)
            elif self.preprocess == 'blur':
                chw = blur(chw)
            chw = stretch(chw, log_ar)
            chw = dataset_transform.inscribed_center_crop(chw)
            chw = transforms.scale(chw, self.scaled_size)
            chw = transforms.center_crop(chw, (self.crop_size, self.crop_size))
            chw = chw.astype(np.float32) / 256.0
            outputs.append(chw)

        return outputs


def stretch(img, log_ar):
    aspect_ratio = np.exp(log_ar)

    height, width = img.shape[1:]
    if aspect_ratio > 1:
        width = int(width * aspect_ratio)
    elif aspect_ratio < 1:
        height = int(height / aspect_ratio)
    else:  # aspect_ratio == 1
        return img

    img = img.transpose(1, 2, 0)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    img = img.transpose(2, 0, 1)
    return img


def blur(img):
    kernel = np.ones((3, 3), np.float32) / 9
    img = img.transpose(1, 2, 0)
    img = cv2.filter2D(img, -1, kernel)
    img = img.transpose(2, 0, 1)
    return img


def edge(img):
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]
                       ], np.float32)
    img = img.transpose(1, 2, 0)
    img = cv2.filter2D(img, -1, kernel)
    img = img.transpose(2, 0, 1)
    return img


def load_test_dataset(filepath, scaled_size, crop_size, log_ars, preprocess):
    dataset = chainer.datasets.ImageDataset(filepath)

    _, test_raw = chainer.datasets.split_dataset(dataset, 17000)
    test_raw, _ = chainer.datasets.split_dataset(test_raw, 100)

    transform = TransformTestdata(scaled_size, crop_size, log_ars, preprocess)
    test = chainer.datasets.TransformDataset(test_raw, transform)
    return test


def load_train_dataset(filepath, scaled_size, crop_size, log_ars):
    dataset = chainer.datasets.ImageDataset(filepath)

    test_raw, _ = chainer.datasets.split_dataset(dataset, 100)

    transform = TransformTestdata(scaled_size, crop_size, log_ars)
    test = chainer.datasets.TransformDataset(test_raw, transform)
    return test


if __name__ == '__main__':
    __spec__ = None  # this line is necessary for multiprocessing on Spyder

    save_root = r'evaluation_results'
    data_file_path = 'E:/voc2012/rgb_jpg_paths_for_paper_v1.3.txt'
    model_files = [
        '0.000312133168336004, 20171206T000238, 4c22664.chainer',  # normal
        '0.00021331777679733932, 20171222T020337, 0b9fee8.chainer',  # random line
        '0.0008955627563409507, 20171213T025642, ac596e6.chainer',  # random blur
        '0.0012712525203824043, 20171220T025634, 0b9fee8.chainer'  # random line & blur
        ]
    batch_size = 100
    scaled_size = 256
    crop_size = 224  # 切り抜きサイズ
    baseline_ar = np.exp(0.12247601469)  # 修正成功とみなすアスペクト比
    ar_max = 3.0
    num_split = 21  # 歪み画像のアスペクト比の段階
    # specify None or 'edge' or 'blur'
    preprocesses = [None, 'edge', 'blur']
#    preprocesses = [None]

    for model_file in model_files:
        # モデル読み込み
        model_path = Path(model_file)
        model = main_resnet.Resnet(
            32, [3, 4, 5, 6], [64, 128, 256, 512], False)
        serializers.load_npz(model_file, model)
        model.to_gpu()
        xp = model.xp

        for preprocess in preprocesses:
            loss_list = []
            loss_abs_list = []
            output_folder_name = Path(model_path.stem)
            if preprocess:
                output_folder_name = output_folder_name / preprocess
            else:
                output_folder_name = output_folder_name / 'no_preprocess'

            # 結果を保存するフォルダを作成
            output_folder_path = Path(save_root) / output_folder_name
            try:
                output_folder_path.mkdir(parents=True)
            except FileExistsError:
                pass  # TODO: remove this try

            log_ars = np.linspace(
                -np.log(ar_max), np.log(ar_max), num_split)
            ds_test = load_test_dataset(
                data_file_path, scaled_size, crop_size, log_ars, preprocess)
            images = np.array([x for x in ds_test]).swapaxes(0, 1)

            ys = []
            with chainer.no_backprop_mode(), chainer.using_config('train',
                                                                  False):
                for x_batch in images:
                    y = model(chainer.cuda.to_gpu(x_batch))
                    ys.append(y.data)
            l_pred = chainer.cuda.to_cpu(xp.stack(ys, 0)).squeeze()
            l_gt = np.broadcast_to(
                log_ars[..., None], (21, 100))  # GT of log AR

            for a in l_pred:
                plt.plot(a)
            plt.show()

            error = l_pred - l_gt
            for a in error:
                plt.plot(a)
            plt.show()

            draw_graph(l_pred, l_gt, baseline_ar, output_folder_path)

            # save success images for non-preprocessed data
            if preprocess is None:
                num_choices = 6
                threshold = np.log(baseline_ar)
                success = np.abs(l_gt - l_pred) < threshold
                success_i_lars, success_i_imgs = success.nonzero()
                choice = np.random.choice(len(success_i_imgs), num_choices)
                success_i_lars = success_i_lars[choice]
                success_i_imgs = success_i_imgs[choice]
                output_failed_image_dir_path = output_folder_path / 'success'
                try:
                    output_failed_image_dir_path.mkdir(parents=True)
                except FileExistsError:
                    pass

                dataset = chainer.datasets.ImageDataset(data_file_path)
                _, test_raw = chainer.datasets.split_dataset(dataset, 17000)
                test_raw, _ = chainer.datasets.split_dataset(test_raw, 100)
                for i_lar, i_img in zip(success_i_lars, success_i_imgs):
                    image = test_raw[i_img]
                    success_lar = l_gt[i_lar, i_img]
                    success_pred = l_pred[i_lar, i_img]
                    image_gt = np.clip(
                        stretch(image, success_lar), 0, 255).astype(np.uint8)
                    image_pred = np.clip(
                        stretch(image, success_lar - success_pred),
                        0, 255).astype(np.uint8)

                    plt.imsave(str(
                        output_failed_image_dir_path / '{}-DI, {}.png'.format(
                            i_img, success_lar)), image_gt.transpose(1, 2, 0))
                    plt.imsave(str(
                        output_failed_image_dir_path / '{}-PR, {}.png'.format(
                            i_img, success_pred)),
                        image_pred.transpose(1, 2, 0))

            # save failed images for non-preprocessed data
            if preprocess is None:
                threshold = np.log(baseline_ar)
                failed = np.abs(l_gt - l_pred) > threshold
                falied_i_lars, failed_i_imgs = failed.nonzero()
                output_failed_image_dir_path = output_folder_path / 'failed'
                try:
                    output_failed_image_dir_path.mkdir(parents=True)
                except FileExistsError:
                    pass

                dataset = chainer.datasets.ImageDataset(data_file_path)
                _, test_raw = chainer.datasets.split_dataset(dataset, 17000)
                test_raw, _ = chainer.datasets.split_dataset(test_raw, 100)
                for i_lar, i_img in zip(falied_i_lars, failed_i_imgs):
                    image = test_raw[i_img]
                    failed_lar = l_gt[i_lar, i_img]
                    failed_pred = l_pred[i_lar, i_img]
                    image_gt = np.clip(
                        stretch(image, failed_lar), 0, 255).astype(np.uint8)
                    image_pred = np.clip(
                        stretch(image, failed_lar - failed_pred),
                        0, 255).astype(np.uint8)

                    plt.imsave(str(
                        output_failed_image_dir_path / '{}-DI, {}.png'.format(
                            i_img, failed_lar)), image_gt.transpose(1, 2, 0))
                    plt.imsave(str(
                        output_failed_image_dir_path / '{}-PR, {}.png'.format(
                            i_img, failed_pred)),
                        image_pred.transpose(1, 2, 0))

        # save test images as sequentially numbered file names
        try:
            output_image_dir_path = Path(save_root) / model_path.stem / 'images'
            output_image_dir_path.mkdir(parents=True)
            with Path(data_file_path).open() as f:
                file_paths = [line.strip() for line in f]
            for i, line in enumerate(file_paths[17000:17100]):
                image = plt.imread(line)
                save_path = output_image_dir_path / '{0:03d}.jpg'.format(i)
                plt.imsave(str(save_path), image)
        except FileExistsError:
            print('SKIP. ("{}" already exists)'.format(
                output_image_dir_path))
