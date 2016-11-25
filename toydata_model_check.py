# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 01:26:22 2016

@author: yamane
"""

import toydata
import numpy as np
from chainer import cuda, optimizers, Chain
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import time
import copy
import tqdm


# ネットワークの定義
class Convnet(Chain):
    def __init__(self, conv1_out_channel=4, conv2_out_channel=4,
                 conv3_out_channel=8, conv4_out_channel=8):
        self.oc1 = conv1_out_channel
        self.oc2 = conv2_out_channel
        self.oc3 = conv3_out_channel
        self.oc4 = conv4_out_channel

        super(Convnet, self).__init__(
            conv1=L.Convolution2D(1, self.oc1, 3, stride=2, pad=1),
            norm1=L.BatchNormalization(self.oc1),
            conv2=L.Convolution2D(self.oc1, self.oc2, 3, stride=2, pad=1),
            norm2=L.BatchNormalization(self.oc2),
            conv3=L.Convolution2D(self.oc2, self.oc3, 3, stride=2, pad=1),
            norm3=L.BatchNormalization(self.oc3),
            conv4=L.Convolution2D(self.oc3, self.oc4, 3, stride=2, pad=1),
            norm4=L.BatchNormalization(self.oc4),

            l1=L.Linear(392, 100),
            norml=L.BatchNormalization(100),
            l2=L.Linear(100, 1),
        )

    def network(self, X, test):
        h = F.relu(self.norm1(self.conv1(X), test=test))
        h = F.relu(self.norm2(self.conv2(h), test=test))
        h = F.relu(self.norm3(self.conv3(h), test=test))
        h = F.relu(self.norm4(self.conv4(h), test=test))
        h = F.relu(self.norml(self.l1(h), test=test))
        y = self.l2(h)
        return y

    def forward(self, X, test):
        y = self.network(X, test)
        return y

    def lossfun(self, X, t, test):
        y = self.forward(X, test)
        loss = F.sigmoid_cross_entropy(y, t)
        accuracy = F.binary_accuracy(y, t)
        return loss, accuracy

    def loss_ave(self, X, T, batch_size, test):
        losses = []
        accuracies = []
        num_data = len(X)
        num_batches = num_data / batch_size
        for indexes in np.array_split(range(num_data), num_batches):
            X_batch = cuda.to_gpu(X[indexes])
            T_batch = cuda.to_gpu(T[indexes])
            loss, accuracy = self.lossfun(X_batch, T_batch, test)
            losses.append(cuda.to_cpu(loss.data))
            accuracies.append(cuda.to_cpu(accuracy.data))
        return np.mean(losses), np.mean(accuracies)

    def __repr__(self):
        template = """conv1_out_channel:{}
conv2_out_channel:{}
conv3_out_channel:{}
conv4_out_channel:{}
conv5_out_channel:{}"""
        return template.format(self.oc1, self.oc2, self.oc3, self.oc4, self.oc5)


if __name__ == '__main__':
    # 超パラメータ
    max_iteration = 100  # 繰り返し回数
    batch_size = 100  # ミニバッチサイズ
    num_train = 1000
    num_valid = 100
    learning_rate = 0.0001
    image_size = 500
    circle_r_min = 50
    circle_r_max = 150
    size_min = 50
    size_max = 200
    p = [0.3, 0.3, 0.4]
    output_size = 112
    aspect_ratio_max = 4
    aspect_ratio_min = 2

    model = Convnet().to_gpu()
    dataset = toydata.RandomCircleSquareDataset(
        image_size, circle_r_min, circle_r_max, size_min, size_max, p,
        output_size, aspect_ratio_max, aspect_ratio_min)
    # Optimizerの設定
    optimizer = optimizers.AdaDelta(learning_rate)
    optimizer.setup(model)

    epoch_loss = []
    epoch_valid_loss = []
    epoch_accuracy = []
    epoch_valid_accuracy = []
    loss_valid_best = np.inf

    num_batches = num_train / batch_size

    time_origin = time.time()
    try:
        for epoch in range(max_iteration):
            time_begin = time.time()
            permu = range(num_train)
            losses = []
            accuracies = []
            for i in tqdm.tqdm(range(num_batches)):
                X_batch, T_batch = dataset.minibatch_binary_classification(
                    batch_size)
                X_batch = cuda.to_gpu(X_batch)
                T_batch = cuda.to_gpu(T_batch)
                # 勾配を初期化
                optimizer.zero_grads()
                # 順伝播を計算し、誤差と精度を取得
                loss, accuracy = model.lossfun(X_batch, T_batch, False)
                # 逆伝搬を計算
                loss.backward()
                optimizer.update()
                losses.append(cuda.to_cpu(loss.data))
                accuracies.append(cuda.to_cpu(accuracy.data))

            time_end = time.time()
            epoch_time = time_end - time_begin
            total_time = time_end - time_origin
            epoch_loss.append(np.mean(losses))
            epoch_accuracy.append(np.mean(accuracies))

            X_valid, T_valid = dataset.minibatch_binary_classification(
                num_valid)
            loss_valid, accuracy_valid = model.loss_ave(
                X_valid, T_valid, batch_size, True)

            epoch_valid_loss.append(loss_valid)
            epoch_valid_accuracy.append(accuracy_valid)

            if loss_valid < loss_valid_best:
                loss_valid_best = loss_valid
                epoch_best = epoch
                model_best = copy.deepcopy(model)

            # 訓練データでの結果を表示
            print "toydata_model_check.py"
            print "epoch:", epoch
            print "time", epoch_time, "(", total_time, ")"
            print "loss[train]:", epoch_loss[epoch]
            print "loss[valid]:", loss_valid
            print "loss[valid_best]:", loss_valid_best
            print "accuracy[train]:", epoch_accuracy[epoch]
            print "accuracy[valid]:", accuracy_valid

            plt.plot(epoch_loss)
            plt.plot(epoch_valid_loss)
            plt.ylim(0, 1)
            plt.title("loss")
            plt.legend(["train", "valid"], loc="upper right")
            plt.grid()
            plt.show()

            plt.plot(epoch_accuracy)
            plt.plot(epoch_valid_accuracy)
            plt.title("accuracy")
            plt.legend(["train", "valid"], loc="lower right")
            plt.grid()
            plt.show()

    except KeyboardInterrupt:
        print "割り込み停止が実行されました"

#    model_filename = 'model' + str(time.time()) + '.npz'
#    serializers.save_npz(model_filename, model_best)

    print 'max_iteration:', max_iteration
    print 'learning_rate:', learning_rate
    print 'batch_size:', batch_size
    print 'train_size', num_train
    print 'valid_size', num_valid
    print dataset
    print model
