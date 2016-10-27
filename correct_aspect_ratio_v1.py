# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:30:18 2016

@author: yamane
"""


import numpy as np
from chainer import cuda, optimizers, Chain, serializers
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
from skimage import color, io, transform
import time
import copy
import tqdm


# ネットワークの定義
class Convnet(Chain):
    def __init__(self):
        super(Convnet, self).__init__(
            conv1=L.Convolution2D(3, 16, 3, stride=2, pad=1),
            conv2=L.Convolution2D(16, 32, 3, stride=1, pad=1),
            conv3=L.Convolution2D(32, 32, 3, stride=2, pad=1),
            conv4=L.Convolution2D(32, 64, 3, stride=1, pad=1),
            conv5=L.Convolution2D(64, 64, 3, stride=2, pad=1),

            l1=L.Linear(254016, 250),
            l2=L.Linear(250, 1),
        )

    def network(self, X):
        h = F.relu(self.conv1(X))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.l1(h))
        y = self.l2(h)
        return y

    def forward(self, X):
        y = self.network(X)
        return y

    def lossfun(self, X, t):
        y = self.forward(X)
        loss = F.sigmoid_cross_entropy(y, t)
        return loss

    def loss_ave(self, image_list, num_batches):
        losses = []
        total_data = np.arange(len(image_list))
        for indexes in np.array_split(total_data, num_batches):
            X_batch, T_batch = read_images_and_T(image_list, indexes)
            loss = self.lossfun(X_batch, T_batch)
            losses.append(cuda.to_cpu(loss.data))
        return np.mean(losses)


def random_aspect_ratio_and_resize_image(image):
    h_image, w_image = image.shape[:2]

    while True:
        aspect_ratio = np.random.rand() * 2  # 0.5~2の乱数を生成
        if aspect_ratio > 0.5:
            break

    w_image = w_image * aspect_ratio

    square_image = transform.resize(image, (500, 500))

    if h_image >= w_image:
        h_image = int(h_image * (500.0 / w_image))
        if (h_image % 2) == 1:
            h_image = h_image + 1
        w_image = 500
        resize_image = transform.resize(image, (h_image, w_image))
        diff = h_image - w_image
        margin = int(diff / 2)
        if margin == 0:
            square_image = resize_image
        else:
            square_image = resize_image[margin:-margin, :]
    else:
        w_image = int(w_image * (500.0 / h_image))
        if (w_image % 2) == 1:
            w_image = w_image + 1
        h_image = 500
        resize_image = transform.resize(image, (h_image, w_image))
        diff = w_image - h_image
        margin = int(diff / 2)
        if margin == 0:
            square_image = resize_image
        else:
            square_image = resize_image[:, margin:-margin]

    return square_image, aspect_ratio


def random_aspect_ratio_and_square_image(image):
    h_image, w_image = image.shape[:2]
    T = 0

    while True:
        aspect_ratio = np.random.rand() * 4  # 0.5~2の乱数を生成
        if aspect_ratio > 0.25:
            break

    square_image = transform.resize(image, (500, 500))

    if np.random.rand() > 0.5:  # 半々の確率で
        w_image = w_image * aspect_ratio
        T = 1

    if h_image >= w_image:
        h_image = int(h_image * (500.0 / w_image))
        if (h_image % 2) == 1:
            h_image = h_image + 1
        w_image = 500
        resize_image = transform.resize(image, (h_image, w_image))
        diff = h_image - w_image
        margin = int(diff / 2)
        if margin == 0:
            square_image = resize_image
        else:
            square_image = resize_image[margin:-margin, :]
    else:
        w_image = int(w_image * (500.0 / h_image))
        if (w_image % 2) == 1:
            w_image = w_image + 1
        h_image = 500
        resize_image = transform.resize(image, (h_image, w_image))
        diff = w_image - h_image
        margin = int(diff / 2)
        if margin == 0:
            square_image = resize_image
        else:
            square_image = resize_image[:, margin:-margin]

    return square_image, T


def read_images_and_T(image_list, indexes):
    images = []
    t = []
    count = 0
    T = np.zeros((len(indexes), 1))

    for i in indexes:
        image = io.imread(image_list[i])
        resize_image, t = random_aspect_ratio_and_square_image(image)
        images.append(resize_image)
        T[count] = t
        count = count + 1
    X = np.stack(images, axis=0)
    X = np.transpose(X, (0, 3, 1, 2))
    X = X.astype(np.float32)
    T = T.astype(np.int32)

    return cuda.to_gpu(X), cuda.to_gpu(T)


if __name__ == '__main__':
    # 超パラメータ
    max_iteration = 1000  # 繰り返し回数
    batch_size = 25  # ミニバッチサイズ
    valid_size = 100
    learning_rate = 0.1

    model = Convnet().to_gpu()
    # Optimizerの設定
    optimizer = optimizers.AdaDelta(learning_rate)
    optimizer.setup(model)

    image_list = []
    epoch_loss = []
    epoch_valid_loss = []
    loss_valid_best = np.inf

    f = open(r"file_list.txt", "r")
    for path in f:
        path = path.strip()
        image_list.append(path)
    f.close()

    train_image_list = image_list[:valid_size]
    valid_image_list = image_list[valid_size:valid_size + 100]

    num_train = len(train_image_list)
    num_batches = num_train / batch_size

    time_origin = time.time()
    try:
        for epoch in range(max_iteration):
            time_begin = time.time()
            permu = range(num_train)
            losses = []
            for indexes in tqdm.tqdm(np.array_split(permu, num_batches)):
                X_batch, T_batch = read_images_and_T(train_image_list, indexes)
                # 勾配を初期化
                optimizer.zero_grads()
                # 順伝播を計算し、誤差と精度を取得
                loss = model.lossfun(X_batch, T_batch)
                # 逆伝搬を計算
                loss.backward()
                optimizer.update()
                losses.append(cuda.to_cpu(loss.data))

            time_end = time.time()
            epoch_time = time_end - time_begin
            total_time = time_end - time_origin
            epoch_loss.append(np.mean(losses))

            loss_valid = model.loss_ave(valid_image_list, num_batches)

            epoch_valid_loss.append(loss_valid)

            if loss_valid < loss_valid_best:
                loss_valid_best = loss_valid
                epoch_best = epoch
                model_best = copy.deepcopy(model)

            # 訓練データでの結果を表示
            print "epoch:", epoch
            print "time", epoch_time, "(", total_time, ")"
            print "loss[train]:", epoch_loss[epoch]
            print "loss[valid]:", loss_valid
            print "loss[valid_best]:", loss_valid_best

            plt.plot(epoch_loss)
            plt.title("loss_train")
            plt.legend(["loss"], loc="upper right")
            plt.grid()
            plt.show()

            plt.plot(epoch_valid_loss)
            plt.title("loss_valid")
            plt.legend(["loss"], loc="upper right")
            plt.grid()
            plt.show()

    except KeyboardInterrupt:
        print "割り込み停止が実行されました"

    model_filename = 'model' + str(time.time()) + '.npz'
    serializers.save_npz(model_filename, model_best)

    print 'max_iteration:', max_iteration
    print 'learning_rate:', learning_rate
    print 'batch_size:', batch_size
    print 'num_train', num_train
