# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:06:11 2017

@author: sakurai
"""

import argparse
from pathlib import Path

import chainer
import numpy as np
import skimage.io
from tqdm import tqdm

import h5py_dataset


def create_voc2012_jpg_paths(dataset_root_dir):
    """
    Args:
        dataset_root_dit (str):
            A str of a directory path.
    """

    root_path = Path(dataset_root_dir)
    output_file_path = root_path / 'voc2012_jpg_paths.txt'
    with output_file_path.open('w') as f:
        jpg_paths = sorted([str(path) for path in root_path.glob('**/*.jpg')])
        f.write('\n'.join(jpg_paths))
    print('A list of jpg files is created to "{}"'.format(output_file_path))
    return output_file_path


if __name__ == '__main__':
    dataset_root_dir = r'E:\voc2012'
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root_dir',
                        help='Specify a directory path where the "VOCdevkit"'
                        ' directory is located.')
    args = parser.parse_args()

    dataset_root_dir = Path(args.dataset_root_dir)
    jpg_paths_file_path = create_voc2012_jpg_paths(dataset_root_dir)
    image_dataset = chainer.datasets.ImageDataset(str(jpg_paths_file_path))

    h5py_path = dataset_root_dir / 'voc2012.hdf5'
    try:
        h5py_dataset.export_as_h5py_dataset_file(
            h5py_path, image_dataset, np.uint8)
    except FileExistsError as e:
        print(e)

    h5py_dataset = h5py_dataset.H5pyDataset(h5py_path)
    print('len(h5py_dataset) == {}'.format(len(h5py_dataset)))
    print('h5py_dataset[0].shape == {}'.format(h5py_dataset[0].shape))
    del h5py_dataset
