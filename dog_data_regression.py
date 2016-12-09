# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 18:37:28 2016

@author: yamane
"""

import numpy as np
import time
import tqdm
import h5py
import copy
import cv2
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
from chainer import cuda, optimizers, Chain, serializers
import chainer.functions as F
import chainer.links as L
import toydata


# ネットワークの定義
class Convnet(Chain):
    def __init__(self):
        super(Convnet, self).__init__(
            conv1=L.Convolution2D(3, 16, 3, stride=2, pad=1),
            norm1=L.BatchNormalization(16),
            conv2=L.Convolution2D(16, 16, 3, stride=2, pad=1),
            norm2=L.BatchNormalization(16),
            conv3=L.Convolution2D(16, 32, 3, stride=2, pad=1),
            norm3=L.BatchNormalization(32),
            conv4=L.Convolution2D(32, 32, 3, stride=2, pad=1),
            norm4=L.BatchNormalization(32),
            conv5=L.Convolution2D(32, 68, 3, stride=2, pad=1),
            norm5=L.BatchNormalization(68),

            l1=L.Linear(3332, 1000),
            norm7=L.BatchNormalization(1000),
            l2=L.Linear(1000, 1),
        )

    def network(self, X, test):
        h = self.conv1(X)
        h = self.norm1(h, test=test)
        h = F.relu(h)
        h = F.relu(self.norm2(self.conv2(h), test=test))
        h = F.relu(self.norm3(self.conv3(h), test=test))
        h = F.relu(self.norm4(self.conv4(h), test=test))
        h = F.relu(self.norm5(self.conv5(h), test=test))
        h = F.relu(self.norm7(self.l1(h), test=test))
        y = self.l2(h)
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
    predict_image = toydata.fix_image(X, predict_r)
    original_image = toydata.fix_image(X, target_r)
    debased_image = np.transpose(X[0], (1, 2, 0))
    predict_image = np.transpose(predict_image, (1, 2, 0))
    original_image = np.transpose(original_image, (1, 2, 0))
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
    plt.imshow(debased_image/256.0)
    plt.subplot(132)
    plt.title("fix_image")
    plt.imshow(predict_image/256.0)
    plt.subplot(133)
    plt.title("target_image")
    plt.imshow(original_image/256.0)
    plt.show()
    return r_loss


def create_mini_batch(queue, file_path, data, batch_size=100, min_ratio=1,
                      max_ratio=4, crop_size=224, output_size=256):
    dataset = h5py.File(file_path)
    image_features = dataset['image_features']
    r_min = min_ratio
    r_max = max_ratio

    num_batches = len(data) / batch_size

    while True:
        for indexes in np.array_split(data, num_batches):
            images = []
            ts = []
            image_batch = image_features[indexes.tolist()]
            for i in range(len(indexes)):
                image = image_batch[i]
                r = toydata.sample_random_aspect_ratio(r_max, r_min)
                image = toydata.change_aspect_ratio(image, r)
                square_image = toydata.crop_center(image)
                resize_image = cv2.resize(square_image,
                                          (output_size, output_size))
                resize_image = toydata.random_crop_and_flip(resize_image,
                                                            crop_size)
                images.append(resize_image)
                t = np.log(r)
                ts.append(t)
            X = np.stack(images, axis=0)
            X = np.transpose(X, (0, 3, 1, 2))
            X = X.astype(np.float32)
            T = np.array(ts, dtype=np.float32).reshape(-1, 1)

            queue.put((X, T))


if __name__ == '__main__':
    # 超パラメータ
    max_iteration = 150  # 繰り返し回数
    batch_size = 100  # ミニバッチサイズ
    num_train = 20000
    num_test = 100
    learning_rate = 0.001
    output_size = 256
    crop_size = 224
    aspect_ratio_min = 1.0
    aspect_ratio_max = 3
    file_path = r'E:\stanford_Dogs_Dataset\raw_dataset_binary\output_size_500\output_size_500.hdf5'

    train_data = range(0, num_train)
    test_data = range(num_train, num_train + num_test)

    queue_train = Queue(10)
    process_train = Process(target=create_mini_batch,
                            args=(queue_train, file_path, train_data,
                                  batch_size, aspect_ratio_min,
                                  aspect_ratio_max, crop_size, output_size))
    process_train.start()
    queue_valid = Queue(10)
    process_valid = Process(target=create_mini_batch,
                            args=(queue_valid, file_path, test_data,
                                  batch_size, aspect_ratio_min,
                                  aspect_ratio_max, crop_size, output_size))
    process_valid.start()
    queue_test = Queue(1)
    process_test = Process(target=create_mini_batch,
                           args=(queue_test, file_path, test_data,
                                 1, aspect_ratio_min, aspect_ratio_max,
                                 crop_size, output_size))
    process_test.start()

    model = Convnet().to_gpu()
    # Optimizerの設定
    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)

    image_list = []
    epoch_loss = []
    epoch_valid_loss = []
    loss_valid_best = np.inf
    r_loss = []

    num_batches_train = num_train / batch_size
    num_batches_test = num_test / batch_size

    time_origin = time.time()
    try:
        for epoch in range(max_iteration):
            time_begin = time.time()
            losses = []
            accuracies = []
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
            print "dog_data_regression.py"
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
            X_test, T_test = queue_test.get()
            r_loss = test_output(model_best, X_test, T_test, r_loss)

    except KeyboardInterrupt:
        print "割り込み停止が実行されました"

    model_filename = 'model_dog_reg' + str(time.time()) + '.npz'
    serializers.save_npz(model_filename, model_best)

    process_train.terminate()
    process_test.terminate()
    print 'max_iteration:', max_iteration
    print 'learning_rate:', learning_rate
    print 'batch_size:', batch_size
    print 'train_size', num_train
    print 'valid_size', num_test
    print 'output_size', output_size
    print 'crop_size', crop_size
    print 'aspect_ratio_min', aspect_ratio_min
    print 'aspect_ratio_max', aspect_ratio_max
