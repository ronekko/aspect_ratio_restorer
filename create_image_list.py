# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 20:24:10 2016

@author: yamane
"""

import os
import numpy as np
from skimage import color, io
import tqdm


if __name__ == '__main__':
    data_location = r'E:\dataset\Places_205'
    dataset_root_dir = r'images'
    root_dir_path = os.path.join(data_location, dataset_root_dir)
    f = open("file_list.txt", "w")
    for root, dirs, files in os.walk(root_dir_path):
        for file_name in tqdm.tqdm(files):
            file_path = os.path.join(root, file_name)
            image = io.imread(file_path)
            if len(image.shape) == 2:
                continue
            f.write(file_path + "\n")
    f.close()
