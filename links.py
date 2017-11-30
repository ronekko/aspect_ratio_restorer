# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 17:33:56 2017

@author: sakurai
"""

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L


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
