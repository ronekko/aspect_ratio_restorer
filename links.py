# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 17:33:56 2017

@author: sakurai
"""

import chainer
import chainer.functions as F
import chainer.links as L

from functions import extend_channels


class Resnet(chainer.Chain):
    '''
    Args:
        n (tuple of ints):
            Numbers of blocks for each stages.
    '''
    def __init__(self, ch_first_conv=32, num_blocks=[3, 3, 4, 4],
                 ch_blocks=[64, 128, 256, 512], use_bottleneck=False):
        self.use_bottleneck = use_bottleneck
        self.first_stage_in_ch = ch_blocks[0]
        if use_bottleneck:
            ch_blocks = [ch * 4 for ch in ch_blocks[1:]]

        n = num_blocks
        ch = ch_blocks
        super(Resnet, self).__init__(
            conv1=L.Convolution2D(3, ch_first_conv, ksize=3, stride=2, pad=1),
            stage2=ResnetStage(n[0], ch[0], True, use_bottleneck),
            stage3=ResnetStage(n[1], ch[1], True, use_bottleneck),
            stage4=ResnetStage(n[2], ch[2], True, use_bottleneck),
            stage5=ResnetStage(n[3], ch[3], True, use_bottleneck),
            bn_out=L.BatchNormalization(ch[3]),
            fc_out=L.Linear(ch[3], 1)
        )

    def __call__(self, x):
        x = self.conv1(x)
        if self.use_bottleneck:
            x = extend_channels(x, self.first_stage_in_ch * 4)
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


class BRCChain(chainer.Chain):
    '''
    This is a composite link of sequence of BatchNormalization, ReLU and
    Convolution2D (a.k.a. pre-activation unit).
    '''
    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None, decay=0.9,
                 **kwargs):
        in_ch, out_ch = in_channels, out_channels
        super(BRCChain, self).__init__(
            bn=L.BatchNormalization(in_ch, decay=decay),
            conv=L.Convolution2D(in_ch, out_ch, ksize=ksize, stride=stride,
                                 pad=pad, nobias=nobias, initialW=initialW,
                                 initial_bias=initial_bias, **kwargs))

    def __call__(self, x):
        x = self.bn(x)
        x = F.relu(x)
        return self.conv(x)


class CBRChain(chainer.Chain):
    '''
    This is a composite link of sequence of Convolution2D, BatchNormalization
    and ReLU.
    '''
    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None, decay=0.9,
                 **kwargs):
        in_ch, out_ch = in_channels, out_channels
        super(CBRChain, self).__init__(
            conv=L.Convolution2D(in_ch, out_ch, ksize=ksize, stride=stride,
                                 pad=pad, nobias=nobias, initialW=initialW,
                                 initial_bias=initial_bias, **kwargs),
            bn=L.BatchNormalization(out_ch, decay=decay))

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class BRPChain(chainer.Chain):
    '''
    This is a composite link of sequence of BatchNormalization, ReLU and
    global AveragePooling2D.
    '''
    def __init__(self, in_channels):
        super(BRPChain, self).__init__(
            bn=L.BatchNormalization(in_channels))

    def __call__(self, x):
        h = self.bn(x)
        h = F.relu(h)
        y = F.average_pooling_2d(h, h.shape[2:])
        return y
