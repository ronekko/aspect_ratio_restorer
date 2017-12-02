# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 15:05:48 2017

@author: ryuhei
"""

import chainer
import chainer.functions as F


def extend_channels(x, out_ch):
    '''Extends channels (i.e. depth) of the input BCHW tensor x by zero-padding
    if out_ch is larger than the number of channels of x, otherwise returns x.
    '''
    b, in_ch, h, w = x.shape
    if in_ch == out_ch:
        return x
    elif in_ch > out_ch:
        raise ValueError('out_ch must be larger than x.shape[1].')

    xp = chainer.cuda.get_array_module(x)
    filler_shape = (b, out_ch - in_ch, h, w)
    filler = xp.zeros(filler_shape, x.dtype)
    return F.concat((x, filler), axis=1)
