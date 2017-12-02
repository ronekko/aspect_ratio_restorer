# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:50:27 2017

@author: sakurai
"""

from types import SimpleNamespace

import chainer
import chainer.functions as F
import chainer.links as L

import common
from functions import extend_channels
from links import BRCChain


class Resnet(chainer.Chain):
    '''
    Args:
        n (int):
            Number of blocks in each group.
    '''
    def __init__(self, n=2, channels=[32, 32, 64, 128, 256],
                 use_bottleneck=False):
        self.use_bottleneck = use_bottleneck
        self.first_stage_in_ch = channels[1]
        if not use_bottleneck:  # default case
            ch_out = channels
        else:
            ch_out = [channels[0]] + [ch * 4 for ch in channels[1:]]

        super(Resnet, self).__init__(
            conv1=L.Convolution2D(3, ch_out[0], ksize=7, stride=2, pad=3),
            stage2=ResnetStage(n, ch_out[1], False, use_bottleneck),
            stage3=ResnetStage(n, ch_out[2], True, use_bottleneck),
            stage4=ResnetStage(n, ch_out[3], True, use_bottleneck),
            stage5=ResnetStage(n, ch_out[4], True, use_bottleneck),
            bn_out=L.BatchNormalization(ch_out[4]),
            fc_out=L.Linear(ch_out[4], 1)
        )

    def __call__(self, x):
        x = self.conv1(x)
        if self.use_bottleneck:
            x = extend_channels(x, self.first_stage_in_ch * 4)
        x = F.max_pooling_2d(x, ksize=3, stride=2)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.bn_out(x)
        x = F.relu(x)
        x = F.average_pooling_2d(x, x.shape[2:])
        return self.fc_out(x)


class ResnetStage(chainer.ChainList):
    '''Sequence of `ResnetBlock`s.
    '''
    def __init__(self, n_blocks, channels, transition, use_bottleneck=True):
        if use_bottleneck:
            block_class = ResnetBottleneckBlock
        else:
            block_class = ResnetBlock

        n_blocks = n_blocks - 1
        blocks = [block_class(channels, transition)]
        blocks += [block_class(channels) for i in range(n_blocks)]
        super(ResnetStage, self).__init__(*blocks)
        self._channels = channels

    def __call__(self, x):
        for block in self:
            x = block(x)
        return x


class ResnetBlock(chainer.Chain):
    '''Residual block (y = x + f(x)) of 'pre-activation'.
    '''
    def __init__(self, ch_out, transition=False):
        self.transition = transition
        ch_in = ch_out // 2 if transition else ch_out
        stride = 2 if transition else 1
        super(ResnetBlock, self).__init__(
            brc1=BRCChain(ch_in, ch_out, 3, stride, pad=1, nobias=True),
            brc2=BRCChain(ch_out, ch_out, 3, pad=1, nobias=True))

    def __call__(self, x):
        h = self.brc1(x)
        h = self.brc2(h)
        if self.transition:
            x = avgpool_and_extend_channels(x)
        return x + h


class ResnetBottleneckBlock(chainer.Chain):
    '''Residual block (y = x + f(x)) of 'pre-activation'.
    '''
    def __init__(self, ch_out, transition=False):
        self.transition = transition
        ch_in = ch_out // 2 if transition else ch_out
        stride = 2 if transition else 1
        bottleneck = ch_out // 4
        super(ResnetBottleneckBlock, self).__init__(
            brc1=BRCChain(ch_in, bottleneck, 1, stride, pad=0, nobias=True),
            brc2=BRCChain(bottleneck, bottleneck, 3, pad=1, nobias=True),
            brc3=BRCChain(bottleneck, ch_out, 1, pad=0, nobias=True))

    def __call__(self, x):
        h = self.brc1(x)
        h = self.brc2(h)
        h = self.brc3(h)
        if self.transition:
            x = avgpool_and_extend_channels(x)
        return x + h


def avgpool_and_extend_channels(x, ch_out=None, ksize=2):
    ch_in = x.shape[1]
    if ch_out is None:
        ch_out = ch_in * 2
    x = F.average_pooling_2d(x, ksize)
    return extend_channels(x, ch_out)


if __name__ == '__main__':
    __spec__ = None  # this line is necessary for multiprocessing on Spyder

    # Hyperparameters
    hparams = SimpleNamespace()
    hparams.filepath = 'E:/voc2012/voc2012.hdf5'
#    hparams.filepath = 'E:/voc2012/rgb_jpg_paths.txt'
#    hparams.filepath = 'E:/voc2012/rgb_jpg_paths_for_paper_v1.3.txt'
    hparams.max_horizontal_factor = 3.0
    hparams.scaled_size = 256
    hparams.crop_size = 224

    # Parameters for network
    use_bottleneck = False
    hparams.n = 4   # number of blocks in each group
    hparams.channels = [32, 32, 64, 128, 256]
#    hparams.channels = [64, 64, 128, 256, 512]

    # Parameters for optimization
    hparams.gpu = 0  # GPU>=0, CPU < 0
    hparams.num_epochs = 2000
    hparams.batch_size = 100
    hparams.lr_init = 0.001
    hparams.optimizer = chainer.optimizers.Adam
#    hparams.weight_decay = 1e-4

    # Model and optimizer
    model = Resnet()

    chainer.config.autotune = True
    result = common.train_eval(model, hparams)

    best_model, best_valid_loss, best_epoch = result[:3]
    train_loss_log, valid_loss_log = result[3:]
