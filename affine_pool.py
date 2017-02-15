# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 20:00:45 2017

@author: yamane
"""

import chainer.functions as F


def pool_affine(model, X, test):
    h = F.relu(model.norm1_1(model.conv1_1(X), test=test))

    h = F.relu(model.norm2_1(model.conv2_1(h), test=test))

    h = F.relu(model.norm3_1(model.conv3_1(h), test=test))

    h = F.relu(model.norm4_1(model.conv4_1(h), test=test))
    h = F.relu(model.norm4_2(model.conv4_2(h), test=test))

    h = F.relu(model.norm5_1(model.conv5_1(h), test=test))
    h = F.relu(model.norm5_2(model.conv5_2(h), test=test))

    h = F.max_pooling_2d(h, 7)

    y = model.l1(h)

    return h, y


def affine_pool(model, conv, X, test):
    h = F.relu(model.norm1_1(model.conv1_1(X), test=test))

    h = F.relu(model.norm2_1(model.conv2_1(h), test=test))

    h = F.relu(model.norm3_1(model.conv3_1(h), test=test))

    h = F.relu(model.norm4_1(model.conv4_1(h), test=test))
    h = F.relu(model.norm4_2(model.conv4_2(h), test=test))

    h = F.relu(model.norm5_1(model.conv5_1(h), test=test))
    h = F.relu(model.norm5_2(model.conv5_2(h), test=test))

    h = conv(h)

    y = F.max_pooling_2d(h, 7)

    return h, y
