# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:29:56 2017

@author: sakurai
"""

from types import SimpleNamespace

import chainer
import chainer.functions as F
import chainer.links as L

import common
from links import CBRChain


class VGGLikeNet(chainer.Chain):
    def __init__(self):
        super(VGGLikeNet, self).__init__(
            cbr1=CBRChain(3, 64, 3, stride=2, pad=1),
            cbr2=CBRChain(64, 128, 3, stride=2, pad=1),
            cbr3=CBRChain(128, 128, 3, stride=2, pad=1),
            cbr41=CBRChain(128, 256, 3, stride=1, pad=1),
            cbr42=CBRChain(256, 256, 3, stride=2, pad=1),
            cbr51=CBRChain(256, 512, 3, stride=1, pad=1),
            cbr52=CBRChain(512, 512, 3, stride=2, pad=1),
            fc=L.Linear(512, 1)
        )

    def __call__(self, x):
        x = self.cbr1(x)
        x = self.cbr2(x)
        x = self.cbr3(x)
        x = self.cbr41(x)
        x = self.cbr42(x)
        x = self.cbr51(x)
        x = self.cbr52(x)
        x = F.max_pooling_2d(x, x.shape[2:])
        y = self.fc(x)
        return y


if __name__ == '__main__':
    # Hyperparameters
    hparams = SimpleNamespace()
    hparams.gpu = 0  # GPU>=0, CPU < 0
    hparams.num_epochs = 2000
    hparams.batch_size = 100
    hparams.lr_init = 0.001
    hparams.optimizer = chainer.optimizers.Adam
#    hparams.weight_decay = 1e-4

    # Model and optimizer
    model = VGGLikeNet()

    chainer.config.autotune = True
    result = common.train_eval(model, hparams)

    best_model, best_valid_loss, best_epoch = result[:3]
    train_loss_log, valid_loss_log = result[3:]
