# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:59:19 2017

@author: sakurai
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import chainer
import chainer.functions as F
from chainer.dataset import concat_examples
from chainer.iterators import MultithreadIterator

from h5py_dataset import H5pyDataset
from load_dataset import Transform


def train_eval(model, hparams):
    p = hparams
    xp = np if p.gpu < 0 else chainer.cuda.cupy

    # Dataset
    filepath = 'E:/tmp_voc2012/voc2012.hdf5'
    max_horizontal_factor = 3.0
    scaled_size = 256
    crop_size = 224
    dataset = H5pyDataset(filepath)
    train_raw, valid_raw = chainer.datasets.split_dataset(dataset, 16500)
    valid_raw, test_raw = chainer.datasets.split_dataset(valid_raw, 500)
    test, _ = chainer.datasets.split_dataset(test_raw, 100)

    transform = Transform(max_horizontal_factor, scaled_size, crop_size)
    train = chainer.datasets.TransformDataset(train_raw, transform)
    valid = chainer.datasets.TransformDataset(valid_raw, transform)
    test = chainer.datasets.TransformDataset(test_raw, transform)
    num_train = len(train)
    num_valid = len(valid)
    num_test = len(test)

#    # normalize by mean and stddev of the train set
#    std_rgb = x_train.std((0, 2, 3), keepdims=True)
#    x_train /= std_rgb
#    x_test /= std_rgb
#    mean_rgb = x_train.mean((0, 2, 3), keepdims=True)
#    x_train -= mean_rgb
#    x_test -= mean_rgb

    # Model and optimizer
    if p.gpu >= 0:
        model.to_gpu()
    optimizer = p.optimizer(p.lr_init)
    optimizer.setup(model)
    if hasattr(p, 'weight_decay') and p.weight_decay > 0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(p.weight_decay))

    # Training loop
    train_loss_log = []
    valid_loss_log = []
    best_valid_loss = np.inf
    try:
        for epoch in range(p.num_epochs):

            epoch_losses = []
            it_train = MultithreadIterator(train, p.batch_size, False, True, 7)
            for batch in tqdm(it_train, total=num_train / p.batch_size):
                # separate images and log aspect ratios
                x_batch, l_batch = concat_examples(batch, p.gpu)
                model.cleargrads()
                with chainer.using_config('train', True):
                    y_batch = model(x_batch)
                    loss = F.mean_squared_error(y_batch, l_batch)
                    loss.backward()
                optimizer.update()
                epoch_losses.append(loss.data)

            epoch_loss = np.mean(chainer.cuda.to_cpu(xp.stack(epoch_losses)))
            train_loss_log.append(epoch_loss)

            # Evaluate the test set
            losses = []
            it_valid = MultithreadIterator(
                valid, p.batch_size, False, False, 7)
            for batch in tqdm(it_valid, total=num_valid / p.batch_size):
                # separate images and log aspect ratios
                x_batch, l_batch = concat_examples(batch, p.gpu)
                with chainer.no_backprop_mode(), \
                        chainer.using_config('train', False):
                    y_batch = model(x_batch)
                    loss = F.mean_squared_error(y_batch, l_batch)
                losses.append(loss.data)
            valid_loss = np.mean(chainer.cuda.to_cpu(xp.stack(losses)))
            valid_loss_log.append(valid_loss)

            # Keep the best model so far
            if valid_loss < best_valid_loss:
                best_model = deepcopy(model)
                best_valid_loss = valid_loss
                best_epoch = epoch

            # Display the training log
            print('{}: loss = {}'.format(epoch, epoch_loss))
            print('valid loss = {}'.format(valid_loss))
            print('best test acc = {} (# {})'.format(best_valid_loss,
                                                     best_epoch))
            print(p)

            plt.figure(figsize=(10, 4))
            plt.title('Loss')
            plt.plot(train_loss_log, label='train loss')
            plt.plot(valid_loss_log, label='valid loss')
            plt.ylim(0, 1)
            plt.legend()
            plt.grid()
            plt.show()

    except KeyboardInterrupt:
        print('Interrupted by Ctrl+c!')

    print('best test acc = {} (# {})'.format(best_valid_loss,
                                             best_epoch))
    print(p)
    print()

    best_model.cleargrads()
    return (best_model, best_valid_loss, best_epoch,
            train_loss_log, valid_loss_log)
