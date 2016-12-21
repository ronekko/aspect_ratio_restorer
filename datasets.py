# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 20:53:05 2016

@author: yamane
"""

import numpy as np
from skimage import draw


class RandomCircleSquareDataset(object):
    def __init__(self, image_size=500, circle_r_min=50, circle_r_max=150,
                 size_min=50, size_max=200, p=[0.3, 0.3, 0.4], output_size=224,
                 aspect_ratio_max=4, aspect_ratio_min=1, batch_size=100):
        self.image_size = image_size
        self.cr_min = circle_r_min
        self.cr_max = circle_r_max
        self.size_min = size_min
        self.size_max = size_max
        self.p = p
        self.output_size = output_size
        self.ar_max = aspect_ratio_max
        self.ar_min = aspect_ratio_min
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def next(self):
        images = []
        for i in range(self.batch_size):
            image = self.create_image()
            images.append(image)
        batch = np.stack(images, axis=0)
        return batch

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
