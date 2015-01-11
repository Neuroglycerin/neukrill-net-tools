#!/usr/bin/env python
"""
Module for all image processing tools
"""

import skimage.transform
import skimage.io


def load_images(image_fpaths, processing, verbose=False):
    """
    Loads images provided in a list of filepaths
    and applies a processing function if supplied one.

    Processing function is expected to take a
    single argument, the image as a numpy array,
    and process it.
    """
    data_subset = []
    num_images = len(image_fpaths)


    for index in range(num_images):
        # read the image into a numpy array
        image = skimage.io.imread(image_fpaths[index])

        if verbose:
            print('image read: {0} of {1}'.format(index, num_images))

        if processing:
            resized_image = processing(image)
            image_vector = resized_image.ravel()
        else:
            image_vector = image.ravel()

        data_subset.append(image_vector)

    return data_subset

def resize(image, size):
    """
    resize images to a pixel*pixel defined in a tuple
    input: image
           size e.g. the tuple (48,48)
    output: resized_image
    """
    resized_image = skimage.transform.resize(image, size)
    return resized_image
