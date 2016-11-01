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
from skimage import transform, draw
import time
import copy
import tqdm


# ネットワークの定義
class Convnet(Chain):
    def __init__(self):
        super(Convnet, self).__init__(
            conv1=L.Convolution2D(1, 64, 3, stride=2, pad=1),
            norm1=L.BatchNormalization(64),
            conv2=L.Convolution2D(64, 64, 3, stride=2, pad=1),
            norm2=L.BatchNormalization(64),
            conv3=L.Convolution2D(64, 128, 3, stride=2, pad=1),
            norm3=L.BatchNormalization(128),
            conv4=L.Convolution2D(128, 128, 3, stride=2, pad=1),
            norm4=L.BatchNormalization(128),
            conv5=L.Convolution2D(128, 256, 3, stride=2, pad=1),
            norm5=L.BatchNormalization(256),
            conv6=L.Convolution2D(256, 256, 3, stride=2, pad=1),
            norm6=L.BatchNormalization(256),

            l1=L.Linear(4096, 1000),
            norm7=L.BatchNormalization(1000),
            l2=L.Linear(1000, 1),
        )

    def network(self, X, test):
        h = F.relu(self.norm1(self.conv1(X), test=test))
        h = F.relu(self.norm2(self.conv2(h), test=test))
        h = F.relu(self.norm3(self.conv3(h), test=test))
        h = F.relu(self.norm4(self.conv4(h), test=test))
        h = F.relu(self.norm5(self.conv5(h), test=test))
        h = F.relu(self.norm6(self.conv6(h), test=test))
        h = F.relu(self.norm7(self.l1(h), test=test))
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

    def loss_ave(self, image_list, num_batches, test):
        losses = []
        accuracies = []
        total_data = np.arange(len(image_list))
        for indexes in np.array_split(total_data, num_batches):
            X_batch, T_batch = read_images_and_T(image_list, indexes)
            loss, accuracy = self.lossfun(X_batch, T_batch, test)
            losses.append(cuda.to_cpu(loss.data))
            accuracies.append(cuda.to_cpu(accuracy.data))
        return np.mean(losses), np.mean(accuracies)


def random_aspect_ratio_and_square_image(image):
    h_image, w_image = image.shape[:2]
    T = 0

    while True:
        aspect_ratio = np.random.rand() * 4  # 0.5~2の乱数を生成
        if aspect_ratio > 0.2 and aspect_ratio < 0.5:
            break
        elif aspect_ratio > 2.0 and aspect_ratio < 4.0:
            break

    square_image = transform.resize(image, (224, 224))

    if np.random.rand() > 0.5:  # 半々の確率で
        w_image = w_image * aspect_ratio
        T = 1

    if h_image >= w_image:
        w_image = int(w_image * (224.0 / h_image))
        if (w_image % 2) == 1:
            w_image = w_image + 1
        h_image = 224
        resize_image = transform.resize(image, (h_image, w_image))
        diff = h_image - w_image
        margin = int(diff / 2)
        if margin == 0:
            square_image = resize_image
        else:
            square_image[:, :margin] = 0
            square_image[:, margin:-margin] = resize_image
            square_image[:, -margin:] = 0
    else:
        h_image = int(h_image * (224.0 / w_image))
        if (h_image % 2) == 1:
            h_image = h_image + 1
        w_image = 224
        resize_image = transform.resize(image, (h_image, w_image))
        diff = w_image - h_image
        margin = int(diff / 2)
        if margin == 0:
            square_image = resize_image
        else:
            square_image[:margin, :] = 0
            square_image[margin:-margin, :] = resize_image
            square_image[-margin:, :] = 0

    return square_image, T


def read_images_and_T(image_list, indexes):
    images = []
    count = 0
    T = np.zeros((len(indexes), 1))

    for i in indexes:
        image = create_image()
        resize_image, t = random_aspect_ratio_and_square_image(image)
        images.append(resize_image)
        T[count] = t
        count = count + 1
    X = np.stack(images, axis=0)
    X = np.transpose(X, (0, 3, 1, 2))
    X = X.astype(np.float32)
    T = T.astype(np.int32)

    return cuda.to_gpu(X), cuda.to_gpu(T)


def create_image():
    image = np.zeros((500, 500), dtype=np.float64)
    rand = np.random.rand()
    x_circle = np.random.randint(150, 350)
    y_circle = np.random.randint(150, 350)
    r_circle = np.random.randint(50, 150)
    x_square = np.random.randint(0, 300)
    y_square = np.random.randint(0, 300)
    r_square = np.random.randint(50, 200)
    if rand > 0.7:  # 半々の確率で
        rr, cc = draw.circle(x_circle, y_circle, r_circle)
        image[rr, cc] = 1
    elif rand > 0.3:
        for i in range(0, r_square):
            rr, cr = draw.line(
                    x_square+i, y_square, x_square+i, y_square+r_square)
            image[rr, cr] = 1
    else:
        rr, cc = draw.circle(x_circle, y_circle, r_circle)
        image[rr, cc] = 1
        for i in range(0, r_square):
            rr, cr = draw.line(
                    x_square+i, y_square, x_square+i, y_square+r_square)
            image[rr, cr] = 1

    image = np.reshape(image, (500, 500, 1))
    return image


if __name__ == '__main__':
    # 超パラメータ
    max_iteration = 1000  # 繰り返し回数
    batch_size = 25  # ミニバッチサイズ
    train_size = 1000
    valid_size = 1000
    learning_rate = 0.0001

    model = Convnet().to_gpu()
    # Optimizerの設定
    optimizer = optimizers.AdaDelta(learning_rate)
    optimizer.setup(model)

    image_list = []
    epoch_loss = []
    epoch_valid_loss = []
    epoch_accuracy = []
    epoch_valid_accuracy = []
    loss_valid_best = np.inf

    f = open(r"file_list.txt", "r")
    for path in f:
        path = path.strip()
        image_list.append(path)
    f.close()

    train_image_list = image_list[:train_size]
    valid_image_list = image_list[-valid_size:]

    num_train = len(train_image_list)
    num_batches = num_train / batch_size

    time_origin = time.time()
    try:
        for epoch in range(max_iteration):
            time_begin = time.time()
            permu = range(num_train)
            losses = []
            accuracies = []
            for indexes in tqdm.tqdm(np.array_split(permu, num_batches)):
                X_batch, T_batch = read_images_and_T(train_image_list, indexes)
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

            loss_valid, accuracy_valid = model.loss_ave(valid_image_list,
                                                        num_batches, False)

            epoch_valid_loss.append(loss_valid)
            epoch_valid_accuracy.append(accuracy_valid)

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
            print "accuracy[train]:", epoch_accuracy[epoch]
            print "accuracy[valid]:", accuracy_valid

            plt.plot(epoch_loss)
            plt.plot(epoch_valid_loss)
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
    print 'num_train', train_size
    print 'num_valid', valid_size
