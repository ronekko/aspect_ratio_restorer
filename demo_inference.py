# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 19:23:17 2018

@author: sakurai
"""

import matplotlib.pyplot as plt

from aspect_ratio_restorer.inference import Restorer


if __name__ == '__main__':
#    image_filename = 'images/2008_002778_ar1.0.jpg'  # original image
#    image_filename = 'images/2008_002778_ar2.0.jpg'  # 2 times wider
    image_filename = 'images/2008_002778_ar0.5.jpg'  # 2 times taller

    restorer = Restorer()

    image = plt.imread(image_filename)

    # If you want to only estimate the aspect ratio of the input image,
    # use `Restorer.predict_aspect_ratio` method.
    aspect_ratio = restorer.predict_aspect_ratio(image)
    print(aspect_ratio)

    # If you want to restore the input image,
    # use `Restorer.restore_image` method.
    restored_image, aspect_ratio = restorer.restore_image(image)
    plt.imshow(restored_image)
    plt.show()
    print('restored_image.shape =', restored_image.shape)
