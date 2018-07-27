# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 19:23:17 2018

@author: sakurai
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import chainer
from chainer.dataset.download import get_dataset_directory, cached_download

from evaluation import stretch
from evaluation import TransformTestdata
import main_resnet


class Restorer(object):

    _model_url = (
        'https://github.com/ronekko/aspect_ratio_restorer/'
        'releases/download/v0.1/'
        'resnet101-20171206T000238-4c22664.npz')

    def __init__(self):
        self.net = main_resnet.Resnet(
            32, [3, 4, 5, 6], [64, 128, 256, 512], False)

        npz_path = download_model(
            self._model_url, 'aspect_ratio_restorer')
        chainer.serializers.load_npz(npz_path, self.net)
        self.preprocessor = Preprocessor()

    def predict_aspect_ratio(self, image):
        preprocessed_array = self.preprocessor.apply(image)
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                log_ar = self.net(preprocessed_array).array
        ar = self.net.xp.exp(log_ar)
        return ar[0, 0]

    def restore_image(self, image, return_aspect_ratio=True):
        ar = self.predict_aspect_ratio(image)
        image = image.transpose(2, 0, 1)
        image = stretch(image, -np.log(ar))  # inverse stratching
        image = image.transpose(1, 2, 0)

        if return_aspect_ratio:
            return image, ar
        else:
            return image


class Preprocessor(object):
    def __init__(self):
        self.transform = TransformTestdata(
            scaled_size=256, crop_size=224, log_ars=[0.0], preprocess=None)

    def apply(self, image):
        """Applies the preprocess for an image.
        Note that an input must be a CHW shaped, RGB ordered,
        [0, 255] valued image.
        """
        return np.array(self.transform(image.transpose(2, 0, 1)))


def download_model(url, subdir_name=None, root_dir_name='ronekko'):
    root_dir_path = get_dataset_directory(root_dir_name)
    basename = os.path.basename(url)
    if subdir_name is None:
        subdir_name = ''
    save_dir_path = os.path.join(root_dir_path, subdir_name)
    save_file_path = os.path.join(save_dir_path, basename)

    if not os.path.exists(save_file_path):
        cache_path = cached_download(url)
        if not os.path.exists(save_dir_path):
            os.mkdir(save_dir_path)
        os.rename(cache_path, save_file_path)
    return save_file_path


if __name__ == '__main__':
#    image_filename = 'images/2008_002778_ar1.0.jpg'  # original image
#    image_filename = 'images/2008_002778_ar2.0.jpg'  # 2 times wider
    image_filename = 'images/2008_002778_ar0.5.jpg'  # 2 times taller

    restorer = Restorer()

    image = plt.imread(image_filename)

    # If you want to only estimate the aspect ratio of the input image,
    # use `Restorer.predict_aspect_ratio` method.
    aspect_ratio = restorer.predict_aspect_ratio(image)
    print(aspect_ratio)

    # If you want to restore the input image,
    # use `Restorer.restore_image` method.
    restored_image, aspect_ratio = restorer.restore_image(image)
    plt.imshow(restored_image)
    plt.show()
    print('restored_image.shape =', restored_image.shape)
