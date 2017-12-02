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
from chainer.iterators import MultiprocessIterator

from h5py_dataset import H5pyDataset
from dataset_transform import Transform


def train_eval(model, hparams):
    p = hparams
    xp = np if p.gpu < 0 else chainer.cuda.cupy

    # Load datasets (as iterators)
    if p.filepath.endswith('.txt'):
        it_train, it_valid, it_test = load_image_dataset_iterators(
            p.filepath, p.batch_size, p.max_horizontal_factor,
            p.scaled_size, p.crop_size)
    elif p.filepath.endswith('.hdf5'):
        it_train, it_valid, it_test = load_h5py_dataset_iterators(
            p.filepath, p.batch_size, p.max_horizontal_factor,
            p.scaled_size, p.crop_size)
    else:
        raise ValueError('"{}" is not supported.'.format(p.filepath))
    num_train = len(it_train.dataset)
    num_valid = len(it_valid.dataset)
    num_test = len(it_test.dataset)

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
            num_batches = int(num_train / p.batch_size)
            for b in tqdm(range(num_batches)):
                batch = next(it_train)
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
            num_batches = int(num_valid / p.batch_size)
            for b in tqdm(range(num_batches)):
                batch = next(it_valid)
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
            print('best valid loss = {} (# {})'.format(best_valid_loss,
                                                       best_epoch))
            print(p)

            plt.figure(figsize=(10, 4))
            plt.title('Loss')
            plt.plot(train_loss_log, label='train loss')
            plt.plot(valid_loss_log, label='valid loss')
            plt.ylim(0, 0.1)
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


def load_image_dataset_iterators(filepath, batch_size, max_horizontal_factor,
                                 scaled_size, crop_size, shuffle_train=True):
    dataset = chainer.datasets.ImageDataset(filepath)

    train_raw, valid_raw = chainer.datasets.split_dataset(dataset, 16500)
    valid_raw, test_raw = chainer.datasets.split_dataset(valid_raw, 500)
    test, _ = chainer.datasets.split_dataset(test_raw, 100)

    transform = Transform(max_horizontal_factor, scaled_size, crop_size)
    train = chainer.datasets.TransformDataset(train_raw, transform)
    valid = chainer.datasets.TransformDataset(valid_raw, transform)
    test = chainer.datasets.TransformDataset(test_raw, transform)

    it_train = MultiprocessIterator(train, batch_size, True, shuffle_train, 5)
    it_valid = MultiprocessIterator(valid, batch_size, True, False, 1, 5)
    it_test = MultiprocessIterator(test, batch_size, True, False, 1, 1)
    return it_train, it_valid, it_test


def load_h5py_dataset_iterators(filepath, batch_size, max_horizontal_factor,
                                scaled_size, crop_size, shuffle_train=True):
    dataset = H5pyDataset(filepath)

    train_raw, valid_raw = chainer.datasets.split_dataset(dataset, 16500)
    valid_raw, test_raw = chainer.datasets.split_dataset(valid_raw, 500)
    test, _ = chainer.datasets.split_dataset(test_raw, 100)

    transform = Transform(max_horizontal_factor, scaled_size, crop_size)
    train = chainer.datasets.TransformDataset(train_raw, transform)
    valid = chainer.datasets.TransformDataset(valid_raw, transform)
    test = chainer.datasets.TransformDataset(test_raw, transform)

    it_train = MultithreadIterator(train, batch_size, True, shuffle_train, 6)
    it_valid = MultithreadIterator(valid, batch_size, True, False, 1)
    it_test = MultithreadIterator(test, batch_size, True, False, 1)
    return it_train, it_valid, it_test
