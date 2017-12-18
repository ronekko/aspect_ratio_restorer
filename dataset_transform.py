# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:58:54 2017

@author: sakurai
"""

import itertools
import math
import random

from chainercv import transforms
import cv2
import numpy as np


class Transform(object):
    """
    An input must be a CHW shaped, RGB ordered, [0, 255] valued image.
    """
    def __init__(self, max_horizontal_factor=4.0,
                 scaled_size=256, crop_size=224,
                 p_blur=0.1, blur_max_ksize=5):
        self.max_horizontal_factor = max_horizontal_factor
        self.scaled_size = scaled_size
        self.crop_size = crop_size
        self.p_blur = p_blur
        self.blur_max_ksize = blur_max_ksize

    def __call__(self, chw):
        chw = chw.astype(np.uint8)
        chw = random_blur(chw, self.p_blur, self.blur_max_ksize)
        chw = transforms.random_flip(chw, False, True)
        chw, param_stretch = random_stretch(chw, self.max_horizontal_factor,
                                            return_param=True)
        chw = inscribed_center_crop(chw)
        chw = transforms.scale(chw, self.scaled_size)
#        chw = transforms.center_crop(chw, (256, 256))
        chw = transforms.random_crop(chw, (self.crop_size, self.crop_size))
        chw = chw.astype(np.float32) / 256.0

        return chw, param_stretch['log_ar'].astype(np.float32)


def random_stretch(img, max_horizontal_factor, max_vertical_factor=None,
                   return_param=False):
    log_ar = sample_log_aspect_ratio(max_horizontal_factor,
                                     max_vertical_factor, 1)
    aspect_ratio = np.exp(log_ar)

    height, width = img.shape[1:]
    if aspect_ratio > 1:
        width = int(width * aspect_ratio)
    elif aspect_ratio < 1:
        height = int(height / aspect_ratio)
    else:  # aspect_ratio == 1
        if return_param:
            return img, {'log_ar': 0.0, 'aspect_ratio': 1.0}
        else:
            return img

    methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA,
               cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
    method = np.random.choice(methods, 1)
    img = img.transpose(1, 2, 0)
    img = cv2.resize(img, (width, height), interpolation=method)
    img = img.transpose(2, 0, 1)

    if return_param:
        return img, {'log_ar': log_ar, 'aspect_ratio': aspect_ratio}
    else:
        return img


def inscribed_center_crop(img, return_param=False, copy=False):
    """Inscribed center crop an image.

    The size of the crop is automatically determined as maximal size, i.e. the
    cropped square has the height or width of the length of shorter side of
    the input image.

    Args:
        img (~numpy.ndarray):
            An image array to be cropped. This is in CHW format.
        return_param (bool):
            If :obj:`True`, this function returns information of slices.
        copy (bool):
            If :obj:`False`, a view of :obj:`img` is returned.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns an array :obj:`out_img` that is cropped from the input
        array.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`out_img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **y_slice** (*slice*): A slice used to crop the input image.\
            The relation below holds together with :obj:`x_slice`.
        * **x_slice** (*slice*): Similar to :obj:`y_slice`.

            .. code::

                out_img = img[:, y_slice, x_slice]

    """
    _, H, W = img.shape
    size = min(H, W)
    oH, oW = size, size

    y_offset = int(round((H - oH) / 2.))
    x_offset = int(round((W - oW) / 2.))

    y_slice = slice(y_offset, y_offset + oH)
    x_slice = slice(x_offset, x_offset + oW)

    img = img[:, y_slice, x_slice]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_slice': y_slice, 'x_slice': x_slice}
    else:
        return img


def sample_log_aspect_ratio(
        max_value, min_value=None, size=None, in_log=False):
    """
    Samples log aspect ratio `l` from a log-uniform distribution:
        `l ~ Uniform(min_value, max_value) if in_log is True`,
        `l ~ Uniform(log(min_value), log(max_value)) if in_log is False`,
    where min_value and max_value are parametesr for lower and upper bounds.
    If min_value is not specified, then it is considered as `-log(max_value)`
    """
    if min_value is None:
        if in_log:
            min_value = -max_value
        else:
            min_value = 1.0 / max_value
    if not in_log:
        max_value = math.log(max_value)
        min_value = math.log(min_value)

    return np.random.uniform(min_value, max_value, size=size)


# TODO: Implement a transform for edge extractor
def random_blur(img, p_blur, max_ksize):
    """Randomly applies random blur. More precisely, blur is applied with
    probability `p_blur` (i.e. it does nothing to the input with probability
    `1 - p_blur`). In addition, the `ksize` of blur is uniformly chosen from a
    set of {3, 5, ..., max_ksize}, i.e. integer that is odd and larger than 3.

    Args:
        img (~numpy.ndarray):
            An image array to add black lines. This is in CHW format.
        p_blur (float):
            The probability that the blur is applied. p_blur \in (0, 1).
        max_ksize (int):
            The max of random `ksize`.
    """

    if random.random() > p_blur:  # the case that does nothing
        return img

    ksize = random.choice(range(3, max_ksize + 1, 2))
    img = img.transpose(1, 2, 0)
    img = cv2.blur(img, (ksize, ksize))
    img = img.transpose(2, 0, 1)
    return img


def add_random_lines(img, p_add, max_num_lines):
    """Add random number of horizontal or vertical black lines.

    Args:
        img (~numpy.ndarray):
            An image array to add black lines. This is in CHW format.
        p_add (float):
            The probability that lines are added. p_add \in (0, 1).
        max_num_lines (int):
            The max number of lines added.
    """

    if random.random() > p_add:  # the case that does nothing
        return img

    n_vertical_lines, n_horizontal_lines = random.choice(
        list(itertools.product(range(max_num_lines + 1),
                               range(max_num_lines + 1))))
    H, W = img.shape[1:]
    y = np.random.choice(H, n_horizontal_lines)
    x = np.random.choice(W, n_vertical_lines)
    img2 = img.copy()
    img2[:, y] = 0
    img2[:, :, x] = 0
    return img2
