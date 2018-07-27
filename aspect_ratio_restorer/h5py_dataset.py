# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 21:06:06 2017

@author: sakurai
"""

from pathlib import Path

import h5py
import numpy as np
import tqdm

import chainer


class H5pyDataset(chainer.dataset.DatasetMixin):
    def __init__(self, filepath):
        self._file = h5py.File(filepath, 'r')
        self._open = True

    def __len__(self):
        return len(self._file['arrays'])

    def get_example(self, i):
        if not self._open:
            raise ValueError('This dataset is no longer available because the'
                             'internal HDF5 file is already closed.')

        arrays = self._file['arrays']
        shapes = arrays.dims[0]['shapes']
        return arrays[i].reshape(shapes[i])

    def close(self):
        self._file.close()

    def __del__(self):
        self.close()


def export_as_h5py_dataset_file(filepath, dataset, dtype, as_hwc=None):
    """
    Exports an instance of a Chainer dataset to an hdf5 file that can be load
    as H5pyDataset. Currently, `chainer.datasets.ImageDataset` is supported.

    Args:
        filepath:
            A str that specifies the file path of the output h5py file.
        dataset:
            A dataset object (e.g. chainer.datasets.ImageDataset) that is
            exported as an h5py file.
        dtype:
            A numpy dtype that specifies the dtype of output h5py data. Note
            that the data in dataset is casted to `dtype`.
        as_hwc (bool):  ## TODO: implement this
            If True, images are stored as (height, width, channels) shaped
            array. If False, images are stored as the same shape.
    """

    filepath = str(filepath)
    if Path(filepath).exists():
        raise FileExistsError('"{}" already exists.'.format(filepath))

    is_tuple_dataset = isinstance(dataset, chainer.datasets.TupleDataset)
    num_examples = len(dataset)

    f = h5py.File(filepath, 'w')

    example = dataset[0]
    if is_tuple_dataset:
        example = chainer.dataset.concat_examples(example)
    vlen_dtype = h5py.special_dtype(vlen=dtype)
    ds_arrays = f.create_dataset(
        'arrays', (num_examples,), vlen_dtype)
    ds_shapes = f.create_dataset(
        'shapes', (num_examples, example.ndim), np.int32)

    for i in tqdm.tqdm(range(num_examples), desc='Creating HDF5 file'):
        array = dataset[i]
        if is_tuple_dataset:
            array = chainer.dataset.concat_examples(array)
        ds_arrays[i] = array.ravel().astype(dtype)
        ds_shapes[i] = array.shape

    ds_arrays.dims.create_scale(ds_shapes, 'shapes')
    ds_arrays.dims[0].attach_scale(ds_shapes)

    f.close()
    print('HDF5 file is created to "{}"'.format(filepath))


if __name__ == '__main__':
    filepath = 'E:/voc2012/voc2012.hdf5'
    h5py_dataset = H5pyDataset(filepath)
