# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 20:00:45 2017

@author: yamane
"""

import chainer.functions as F


def pool_affine(model, X, test):
    h = F.relu(model.norm1(model.conv1(X), test=test))
    h = F.relu(model.norm2(model.conv2(h), test=test))
    h = F.relu(model.norm3(model.conv3(h), test=test))
    h = F.relu(model.norm4(model.conv4(h), test=test))
    h = F.relu(model.norm5(model.conv5(h), test=test))
    h = F.average_pooling_2d(h, 7)
    y = model.l1(h)
    return h, y


def affine_pool(model, conv, X, test):
    h = F.relu(model.norm1(model.conv1(X), test=test))
    h = F.relu(model.norm2(model.conv2(h), test=test))
    h = F.relu(model.norm3(model.conv3(h), test=test))
    h = F.relu(model.norm4(model.conv4(h), test=test))
    h = F.relu(model.norm5(model.conv5(h), test=test))
    h = conv(h)
    y = F.average_pooling_2d(h, 7)
    return h, y
