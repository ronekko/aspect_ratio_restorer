# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 21:45:49 2016

@author: yamane
"""

import os
import numpy as np
import time
import copy
import tqdm
import matplotlib.pyplot as plt
from chainer import cuda, optimizers, Chain, serializers
import chainer.functions as F
import chainer.links as L
import toydata


# ネットワークの定義
class Convnet(Chain):
    def __init__(self):
        super(Convnet, self).__init__(
            conv1=L.Convolution2D(1, 16, 3, stride=2, pad=1),
            norm1=L.BatchNormalization(16),
            conv2=L.Convolution2D(16, 16, 3, stride=2, pad=1),
            norm2=L.BatchNormalization(16),
            conv3=L.Convolution2D(16, 32, 3, stride=2, pad=1),
            norm3=L.BatchNormalization(32),
            conv4=L.Convolution2D(32, 32, 3, stride=2, pad=1),
            norm4=L.BatchNormalization(32),
            conv5=L.Convolution2D(32, 64, 3, stride=2, pad=1),
            norm5=L.BatchNormalization(64),

            l1=L.Linear(3136, 1000),
            norm6=L.BatchNormalization(1000),
            l2=L.Linear(1000, 1),
        )

    def network(self, X, test):
        h = F.relu(self.norm1(self.conv1(X), test=test))
        h = F.relu(self.norm2(self.conv2(h), test=test))
        h = F.relu(self.norm3(self.conv3(h), test=test))
        h = F.relu(self.norm4(self.conv4(h), test=test))
        h = F.relu(self.norm5(self.conv5(h), test=test))
        h = F.tanh(self.norm6(self.l1(h), test=test))
        y = self.l2(h)
        return y

    def forward(self, X, test):
        y = self.network(X, test)
        return y

    def lossfun(self, X, t, test):
        y = self.forward(X, test)
        loss = F.mean_squared_error(y, t)
        return loss

    def loss_ave(self, X, T, batch_size, test):
        losses = []
        num_data = len(X)
        num_batches = num_data / batch_size
        for indexes in np.array_split(range(num_data), num_batches):
            X_batch = cuda.to_gpu(X[indexes])
            T_batch = cuda.to_gpu(T[indexes])
            loss = self.lossfun(X_batch, T_batch, test)
            losses.append(cuda.to_cpu(loss.data))
        return np.mean(losses)

    def predict(self, X, test):
        X = cuda.to_gpu(X)
        y = self.forward(X, test)
        y = cuda.to_cpu(y.data)
        return y


if __name__ == '__main__':
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    output_location = 'C:\Users\yamane\Dropbox\correct_aspect_ratio'
    output_root_dir = os.path.join(output_location, file_name)
    output_root_dir = os.path.join(output_root_dir, str(time.time()))

    if os.path.exists(output_root_dir):
        pass
    else:
        os.makedirs(output_root_dir)

    # 超パラメータ
    max_iteration = 100  # 繰り返し回数
    batch_size = 100  # ミニバッチサイズ
    num_train = 5000
    num_valid = 100
    learning_rate = 0.001
    image_size = 500
    circle_r_min = 50
    circle_r_max = 150
    size_min = 50
    size_max = 200
    p = [0.3, 0.3, 0.4]
    output_size = 224
    aspect_ratio_min = 1.0
    aspect_ratio_max = 3.0

    model = Convnet().to_gpu()
    dataset = toydata.RandomCircleSquareDataset(
        image_size, circle_r_min, circle_r_max, size_min, size_max, p,
        output_size, aspect_ratio_max, aspect_ratio_min)
    # Optimizerの設定
    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)

    epoch_loss = []
    epoch_valid_loss = []
    r_loss = []
    loss_valid_best = np.inf

    num_batches = num_train / batch_size

    time_origin = time.time()
    try:
        for epoch in range(max_iteration):
            time_begin = time.time()
            losses = []
            for i in tqdm.tqdm(range(num_batches)):
                X_batch, T_batch = dataset.minibatch_regression(batch_size)
                X_batch = cuda.to_gpu(X_batch)
                T_batch = cuda.to_gpu(T_batch)
                # 勾配を初期化
                optimizer.zero_grads()
                # 順伝播を計算し、誤差と精度を取得
                loss = model.lossfun(X_batch, T_batch, False)
                # 逆伝搬を計算
                loss.backward()
                optimizer.update()
                losses.append(cuda.to_cpu(loss.data))

            time_end = time.time()
            epoch_time = time_end - time_begin
            total_time = time_end - time_origin
            epoch_loss.append(np.mean(losses))

            X_valid, T_valid = dataset.minibatch_regression(num_valid)
            loss_valid = model.loss_ave(X_valid, T_valid, batch_size, True)

            # テスト用のデータを取得
            X_test, T_test = dataset.minibatch_regression(1)
            predict_t = model.predict(X_test, True)
            target_t = T_test
            predict_r = np.exp(predict_t)
            target_r = np.exp(target_t)
            predict_image = toydata.fix_image(X_test, predict_r)
            original_image = toydata.fix_image(X_test, target_r)

            r_dis = np.absolute(predict_r - target_r)
            r_loss.append(r_dis[0])

            epoch_valid_loss.append(loss_valid)

            if loss_valid < loss_valid_best:
                loss_valid_best = loss_valid
                model_best = copy.deepcopy(model)

            # 訓練データでの結果を表示
            print "toydata_regression.py"
            print "epoch:", epoch
            print "time", epoch_time, "(", total_time, ")"
            print "loss[train]:", epoch_loss[epoch]
            print "loss[valid]:", loss_valid
            print "loss[valid_best]:", loss_valid_best
            print 'predict t:', predict_t, 'target t:', target_t
            print 'predict r:', predict_r, 'target r:', target_r

            plt.plot(epoch_loss)
            plt.plot(epoch_valid_loss)
            plt.ylim(0, 0.5)
            plt.title("loss")
            plt.legend(["train", "valid"], loc="upper right")
            plt.grid()
            plt.show()

            plt.plot(r_loss)
            plt.title("r_disdance")
            plt.grid()
            plt.show()

            plt.subplot(131)
            plt.title("debased_image")
            plt.imshow(X_test[0][0])
            plt.subplot(132)
            plt.title("fix_image")
            plt.imshow(predict_image[0])
            plt.subplot(133)
            plt.title("target_image")
            plt.imshow(original_image[0])
            plt.show()

    except KeyboardInterrupt:
        print "割り込み停止が実行されました"

    model_filename = str(file_name) + str(time.time()) + '.npz'
    loss_filename = 'epoch_loss' + str(time.time()) + '.png'
    r_dis_filename = 'r_distance' + str(time.time()) + '.png'
    model_filename = os.path.join(output_root_dir, model_filename)
    loss_filename = os.path.join(output_root_dir, loss_filename)
    r_dis_filename = os.path.join(output_root_dir, r_dis_filename)

    plt.plot(epoch_loss)
    plt.plot(epoch_valid_loss)
    plt.ylim(0, 0.5)
    plt.title("loss")
    plt.legend(["train", "valid"], loc="upper right")
    plt.grid()
    plt.savefig(loss_filename)
    plt.show()

    plt.plot(r_loss)
    plt.title("r_disdance")
    plt.grid()
    plt.savefig(r_dis_filename)
    plt.show()

    model_filename = os.path.join(output_root_dir, model_filename)
    serializers.save_npz(model_filename, model_best)

    print 'max_iteration:', max_iteration
    print 'learning_rate:', learning_rate
    print 'batch_size:', batch_size
    print 'train_size', num_train
    print 'valid_size', num_valid
    print dataset
