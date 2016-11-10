# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 20:24:10 2016

@author: yamane
"""

import os
from skimage import io
import tqdm


if __name__ == '__main__':
    data_location = r'E:\stanford_Dogs_Dataset\Images'
    f = open("file_list.txt", "w")
    for root, dirs, files in os.walk(data_location):
        for file_name in tqdm.tqdm(files):
            file_path = os.path.join(root, file_name)
            image = io.imread(file_path)
            if len(image.shape) == 2:
                continue
            elif len(image[0][0]) != 3:
                continue
            f.write(file_path + "\n")
    f.close()
