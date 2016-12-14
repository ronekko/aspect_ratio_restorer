# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 20:53:05 2016

@author: yamane
"""

import numpy as np
import cv2
from skimage import draw
import h5py
import utility


class RandomCircleSquareDataset(object):
    def __init__(self, image_size=500, circle_r_min=50, circle_r_max=150,
                 size_min=50, size_max=200, p=[0.3, 0.3, 0.4], output_size=224,
                 aspect_ratio_max=4, aspect_ratio_min=1):
        self.image_size = image_size
        self.cr_min = circle_r_min
        self.cr_max = circle_r_max
        self.size_min = size_min
        self.size_max = size_max
        self.p = p
        self.output_size = output_size
        self.ar_max = aspect_ratio_max
        self.ar_min = aspect_ratio_min

    def minibatch_binary_classification(self, batch_size):
        images = []
        ts = []

        for i in range(batch_size):
            image = self.create_image()
            t = np.random.choice(2)
            if t == 1:
                r = utility.sample_random_aspect_ratio(
                    self.ar_max, self.ar_min)
            else:
                r = 1
            image = utility.change_aspect_ratio(image, r)
            square_image = utility.crop_center(image)
            resize_image = cv2.resize(
                square_image, (self.output_size, self.output_size))
            resize_image = resize_image[..., None]
            images.append(resize_image)
            ts.append(t)
        X = np.stack(images, axis=0)
        X = np.transpose(X, (0, 3, 1, 2))
        X = X.astype(np.float32)
        T = np.array(ts, dtype=np.int32).reshape(-1, 1)

        return X, T

    def minibatch_regression(self, batch_size):
        images = []
        ts = []

        for i in range(batch_size):
            image = self.create_image()
            r = utility.sample_random_aspect_ratio(self.ar_max, self.ar_min)
            image = utility.change_aspect_ratio(image, r)
            square_image = utility.padding_image(image)
            # cv2.resize:(image, (w, h))
            # transform.resize:(image, (h, w))
            resize_image = cv2.resize(
                square_image, (self.output_size, self.output_size))
            resize_image = resize_image[..., None]
            images.append(resize_image)
            t = np.log(r)
            ts.append(t)
        X = np.stack(images, axis=0)
        X = np.transpose(X, (0, 3, 1, 2))
        X = X.astype(np.float32)
        T = np.array(ts, dtype=np.float32).reshape(-1, 1)

        return X, T

    def test_regression(self, batch_size, image, r):
        images = []
        ts = []

        for i in range(batch_size):
            image = utility.change_aspect_ratio(image, r)
            square_image = utility.padding_image(image)
            # cv2.resize:(image, (w, h))
            # transform.resize:(image, (h, w))
            resize_image = cv2.resize(
                square_image, (self.output_size, self.output_size))
            resize_image = resize_image[..., None]
            images.append(resize_image)
            t = np.log(r)
            ts.append(t)
        X = np.stack(images, axis=0)
        X = np.transpose(X, (0, 3, 1, 2))
        X = X.astype(np.float32)
        T = np.array(ts, dtype=np.float32).reshape(-1, 1)
        return X, T

    def create_image(self):
        case = np.random.choice(3, p=self.p)
        if case == 0:
            image = self.create_random_circle(
                self.image_size, self.cr_min, self.cr_max)
        elif case == 1:
            image = self.create_random_square(
                self.image_size, self.size_min, self.size_max)
        else:
            image = self.create_random_circle_square(
                self.image_size, self.cr_min, self.cr_max,
                self.size_min, self.size_max)
        return image

    def create_random_circle(self, image_size, r_min, r_max):
        image = np.zeros((image_size, image_size), dtype=np.float64)
        r = np.random.randint(r_min, r_max)
        x = np.random.randint(r-1, image_size - r + 1)
        y = np.random.randint(r-1, image_size - r + 1)

        rr, cc = draw.circle(x, y, r)
        image[rr, cc] = 1

        image = np.reshape(image, (image_size, image_size, 1))
        return image

    def create_random_square(self, image_size, size_min, size_max):
        image = np.zeros((image_size, image_size), dtype=np.float64)
        size = np.random.randint(size_min, size_max)
        x = np.random.randint(0, image_size-size+1)
        y = np.random.randint(0, image_size-size+1)

        for i in range(0, size):
            rr, cr = draw.line(y, x+i, y+size-1, x+i)
            image[rr, cr] = 1

        image = np.reshape(image, (image_size, image_size, 1))
        return image

    def create_random_circle_square(
            self, image_size, r_min, r_max, size_min, size_max):
        circle = self.create_random_circle(image_size, r_min, r_max)
        square = self.create_random_square(image_size, size_min, size_max)
        image = np.logical_or(circle, square)
        image = image.astype(np.float64)
        return image

    def __repr__(self):
        template = """image_size:{}
output_size:{}
circle_min:{}
circle_max:{}
size_min:{}
size_max:{}
p:{}
aspect_ratio_min:{}
aspect_ratio_max:{}"""
        return template.format(self.image_size, self.output_size, self.cr_min,
                               self.cr_max, self.size_min, self.size_max,
                               self.p, self.ar_min, self.ar_max)


class DogDataset(object):
    def __init__(self, hdf5file_path, output_size=256, crop_size=224,
                 aspect_ratio_max=3, aspect_ratio_min=1):
        self.file_path = hdf5file_path
        self.dataset = h5py.File(hdf5file_path)
        self.image_features = self.dataset['image_features']
        self.output_size = output_size
        self.crop_size = crop_size
        self.ar_max = aspect_ratio_max
        self.ar_min = aspect_ratio_min

    def minibatch_classification(self, queue, data, batch_size=100):

        num_batches = len(data) / batch_size

        while True:
            for indexes in np.array_split(data, num_batches):
                images = []
                ts = []
                image_batch = self.image_features[indexes.tolist()]
                for i in range(len(indexes)):
                    image = image_batch[i]
                    t = np.random.choice(2)
                    if t == 1:
                        r = utility.sample_random_aspect_ratio(self.ar_max,
                                                               self.ar_min)
                    else:
                        r = 1
                    image = utility.change_aspect_ratio(image, r)
                    square_image = utility.crop_center(image)
                    resize_image = cv2.resize(
                        square_image, (self.output_size, self.output_size))
                    resize_image = utility.random_crop_and_flip(resize_image,
                                                                self.crop_size)
                    images.append(resize_image)
                    ts.append(t)
                X = np.stack(images, axis=0)
                X = np.transpose(X, (0, 3, 1, 2))
                X = X.astype(np.float32)
                T = np.array(ts, dtype=np.int32).reshape(-1, 1)

                queue.put((X, T))

    def minibatch_regression(self, queue, data, batch_size=100):

        num_batches = len(data) / batch_size

        while True:
            for indexes in np.array_split(data, num_batches):
                images = []
                ts = []
                image_batch = self.image_features[indexes.tolist()]
                for i in range(len(indexes)):
                    image = image_batch[i]
                    r = utility.sample_random_aspect_ratio(self.ar_max,
                                                           self.ar_min)
                    image = utility.change_aspect_ratio(image, r)
                    square_image = utility.crop_center(image)
                    resize_image = cv2.resize(
                        square_image, (self.output_size, self.output_size))
                    resize_image = utility.random_crop_and_flip(resize_image,
                                                                self.crop_size)
                    images.append(resize_image)
                    t = np.log(r)
                    ts.append(t)
                X = np.stack(images, axis=0)
                X = np.transpose(X, (0, 3, 1, 2))
                X = X.astype(np.float32)
                T = np.array(ts, dtype=np.float32).reshape(-1, 1)

                queue.put((X, T))

    def test_regression(self, hdf5file_path, data, batch_size, r):
        dataset = h5py.File(hdf5file_path)
        image_features = dataset['image_features']

        num_batches = len(data) / batch_size

        for indexes in np.array_split(data, num_batches):
            images = []
            ts = []
            image_batch = image_features[indexes.tolist()]
            for i in range(len(indexes)):
                image = image_batch[i]
                image = utility.change_aspect_ratio(image, r)
                square_image = utility.crop_center(image)
                resize_image = cv2.resize(square_image,
                                          (self.output_size, self.output_size))
                resize_image = utility.random_crop_and_flip(resize_image,
                                                            self.crop_size)
                images.append(resize_image)
                t = np.log(r)
                ts.append(t)
            X = np.stack(images, axis=0)
            X = np.transpose(X, (0, 3, 1, 2))
            X = X.astype(np.float32)
            T = np.array(ts, dtype=np.float32).reshape(-1, 1)
            return X, T

    def __repr__(self):
        template = """output_size:{}
crop_size:{}
aspect_ratio_min:{}
aspect_ratio_max:{}"""
        return template.format(self.output_size, self.crop_size,
                               self.ar_min, self.ar_max)
