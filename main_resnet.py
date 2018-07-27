# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:50:27 2017

@author: sakurai
"""

import time
from types import SimpleNamespace

import chainer
import chainer.functions as F
import chainer.links as L

import common
from links import Resnet


if __name__ == '__main__':
    __spec__ = None  # this line is necessary for multiprocessing on Spyder

    # Hyperparameters
    hparams = SimpleNamespace()
#    hparams.filepath = 'E:/voc2012/voc2012.hdf5'
#    hparams.filepath = 'E:/voc2012/rgb_jpg_paths.txt'
    hparams.filepath = 'E:/voc2012/rgb_jpg_paths_for_paper_v1.3.txt'
    hparams.max_horizontal_factor = 3.0
    hparams.scaled_size = 256
    hparams.crop_size = 224
    hparams.p_blur = 0.3
    hparams.blur_max_ksize = 3
    hparams.p_add_lines = 0.3
    hparams.max_num_lines = 2

    # Parameters for network
    hparams.ch_first_conv = 32
    hparams.num_blocks = [3, 4, 5, 6]
    hparams.ch_blocks = [64, 128, 256, 512]
    hparams.use_bottleneck = False

    # Parameters for optimization
    hparams.gpu = 0  # GPU>=0, CPU < 0
    hparams.num_epochs = 2000
    hparams.batch_size = 100
    hparams.lr_init = 0.001
    hparams.optimizer = chainer.optimizers.Adam
#    hparams.weight_decay = 1e-4

    # Model and optimizer
    model = Resnet(hparams.ch_first_conv, hparams.num_blocks,
                   hparams.ch_blocks, hparams.use_bottleneck)

    chainer.config.autotune = True
    result = common.train_eval(model, hparams)

    best_model, best_valid_loss, best_epoch = result[:3]
    train_loss_log, valid_loss_log = result[3:]

    model_file_name = '{}, {}.chainer'.format(best_valid_loss,
                                              time.strftime("%Y%m%dT%H%M%S"))
    chainer.serializers.save_npz(model_file_name, model)
