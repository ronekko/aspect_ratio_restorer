# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 13:51:10 2016

@author: yamane
"""

import os
import numpy as np
import time
import copy
import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue

from chainer import cuda, optimizers, Chain, serializers
import chainer.functions as F
import chainer.links as L

import utility
import load_datasets


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

            norm6=L.BatchNormalization(64),
            l1=L.Linear(64, 1),
        )

    def network(self, X, test):
        h = F.relu(self.norm1(self.conv1(X), test=test))
        h = F.relu(self.norm2(self.conv2(h), test=test))
        h = F.relu(self.norm3(self.conv3(h), test=test))
        h = F.relu(self.norm4(self.conv4(h), test=test))
        h = F.relu(self.norm5(self.conv5(h), test=test))
        h = F.relu(self.norm6(F.max_pooling_2d(h, 7), test=test))
        y = self.l1(h)
        return y

    def forward(self, X, test):
        y = self.network(X, test)
        return y

    def lossfun(self, X, t, test):
        y = self.forward(X, test)
        loss = F.mean_squared_error(y, t)
        return loss

    def loss_ave(self, queue, num_batches, test):
        losses = []
        for i in range(num_batches):
            X_batch, T_batch = queue.get()
            X_batch = cuda.to_gpu(X_batch)
            T_batch = cuda.to_gpu(T_batch)
            loss = self.lossfun(X_batch, T_batch, test)
            losses.append(cuda.to_cpu(loss.data))
        return np.mean(losses)

    def predict(self, X, test):
        X = cuda.to_gpu(X)
        y = self.forward(X, test)
        y = cuda.to_cpu(y.data)
        return y


def test_output(model, X, T, r_loss):
    predict_t = model.predict(X, True)
    target_t = T
    predict_r = np.exp(predict_t)
    target_r = np.exp(target_t)
    predict_image = utility.fix_image(X, predict_r)
    original_image = utility.fix_image(X, target_r)
    r_dis = np.absolute(predict_r - target_r)
    r_loss.append(r_dis[0])

    print 'predict t:', predict_t, 'target t:', target_t
    print 'predict r:', predict_r, 'target r:', target_r

    plt.plot(r_loss)
    plt.title("r_disdance")
    plt.grid()
    plt.show()

    plt.subplot(131)
    plt.title("debased_image")
    plt.imshow(X[0][0])
    plt.subplot(132)
    plt.title("fix_image")
    plt.imshow(predict_image[0])
    plt.subplot(133)
    plt.title("target_image")
    plt.imshow(original_image[0])
    plt.show()
    return r_loss


if __name__ == '__main__':
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    time_start = time.time()
    epoch_loss = []
    epoch_valid_loss = []
    r_loss = []
    loss_valid_best = np.inf
    # 超パラメータ
    max_iteration = 100  # 繰り返し回数
    batch_size = 100  # ミニバッチサイズ
    num_train = 5000
    num_test = 100
    learning_rate = 0.001
    image_size = 500
    circle_r_min = 50
    circle_r_max = 150
    size_min = 50
    size_max = 200
    p = [0.3, 0.3, 0.4]
    crop_size = 224
    aspect_ratio_min = 1.0
    aspect_ratio_max = 2.0
    crop = False
    output_location = 'C:\Users\yamane\Dropbox\correct_aspect_ratio'

    output_root_dir = os.path.join(output_location, file_name)
    folder_name = str(time_start) + '_asp_max_' + str(aspect_ratio_max)
    output_root_dir = os.path.join(output_root_dir, folder_name)
    if os.path.exists(output_root_dir):
        pass
    else:
        os.makedirs(output_root_dir)
    model_filename = str(file_name) + str(time.time()) + '.npz'
    loss_filename = 'epoch_loss' + str(time.time()) + '.png'
    r_dis_filename = 'r_distance' + str(time.time()) + '.png'
    model_filename = os.path.join(output_root_dir, model_filename)
    loss_filename = os.path.join(output_root_dir, loss_filename)
    r_dis_filename = os.path.join(output_root_dir, r_dis_filename)
    # バッチサイズ計算
    num_batches_train = num_train / batch_size
    num_batches_test = num_test / batch_size
    # stream作成
    toy_stream_train, toy_stream_test = load_datasets.load_toy_stream(batch_size)
    # キューを作成、プロセススタート
    queue_train = Queue(10)
    process_train = Process(target=load_datasets.load_data,
                            args=(queue_train, toy_stream_train, crop,
                                  aspect_ratio_max, aspect_ratio_min,
                                  crop_size))
    process_train.start()
    queue_valid = Queue(10)
    process_valid = Process(target=load_datasets.load_data,
                            args=(queue_valid, toy_stream_test, crop,
                                  aspect_ratio_max, aspect_ratio_min,
                                  crop_size))
    process_valid.start()
    # モデル読み込み
    model = Convnet().to_gpu()
    # Optimizerの設定
    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)

    time_origin = time.time()
    try:
        for epoch in range(max_iteration):
            time_begin = time.time()
            losses = []
            for i in tqdm.tqdm(range(num_batches_train)):
                X_batch, T_batch = queue_train.get()
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

            loss_valid = model.loss_ave(queue_valid, num_batches_test, True)
            epoch_valid_loss.append(loss_valid)
            if loss_valid < loss_valid_best:
                loss_valid_best = loss_valid
                epoch__loss_best = epoch
                model_best = copy.deepcopy(model)

            # 訓練データでの結果を表示
            print "toydata_regression_max_pooling.py"
            print "epoch:", epoch
            print "time", epoch_time, "(", total_time, ")"
            print "loss[train]:", epoch_loss[epoch]
            print "loss[valid]:", loss_valid
            print "loss[valid_best]:", loss_valid_best

            plt.plot(epoch_loss)
            plt.plot(epoch_valid_loss)
            plt.ylim(0, 0.5)
            plt.title("loss")
            plt.legend(["train", "valid"], loc="upper right")
            plt.grid()
            plt.show()

            # テスト用のデータを取得
            X_test, T_test = queue_valid.get()
            r_loss = test_output(model_best, X_test[0:1], T_test[0:1], r_loss)

    except KeyboardInterrupt:
        print "割り込み停止が実行されました"

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

    process_train.terminate()
    process_valid.terminate()
    print 'max_iteration:', max_iteration
    print 'learning_rate:', learning_rate
    print 'batch_size:', batch_size
    print 'train_size', num_train
    print 'valid_size', num_test
