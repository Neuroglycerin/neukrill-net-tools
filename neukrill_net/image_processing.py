#!/usr/bin/env python
"""
Module for all image processing tools
"""

import skimage.transform
import skimage.io
import skimage.util

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
            # make sure that what we get out is a list
            # even if it's a single image
            resized_images = processing(image)
            if type(resized_images) is not list:
                resized_images = [resized_images]
            image_vectors = list(map(lambda image: image.ravel(),
                                            resized_images))
        else:
            image_vectors = [image.ravel()]

        data_subset += image_vectors

    return data_subset

def resize_image(image, size):
    """
    resize images to a pixel*pixel defined in a tuple
    input: image
           size e.g. the tuple (48,48)
    output: resized_image

    does this by padding to make the image square, then
    resizing
    """
    # find out how much we have to pad
    diff = abs(image.shape[1]-image.shape[0])
    # check if difference is divisible by two
    if diff%2 > 0:
        # not divisible, add extra pixel to one side
        extra = 1
        # then split to add to each side
        d = int((diff-1)/2)
    else:
        extra = 0
        d = int(diff/2)
    # which side to pad?
    if image.shape[1] > image.shape[0]:
        # more columns than rows, pad the rows
        padded_image = skimage.util.pad(image, ((d,d+extra),(0,0)),
                                            'edge')
    else:
        # more rows than columns, pad the columns
        padded_image = skimage.util.pad(image, ((0,0),(d+extra,d)),
                                            'edge')
    if padded_image.shape[0] != padded_image.shape[1]:
        raise ValueError("Padded image is not square"
                            "Raise an issue about this."
                            "print rows: {0}, columns:{1}".format(
                        padded_image.shape[0],padded_image.shape[1]))

    # now resize to specified size
    resized_image = skimage.transform.resize(padded_image, size)

    return resized_image

def flip_image(image, flip_x=False, flip_y=False):
    """
    Flips 2D images in either X or Y axis or both axes.
    Non-destructive: returns a copy of the input image
    input:  image  - input image
            flip_x - whether to flip image in X axis
            flip_y - whether to flip image in Y axis
    output: flipped_image - a new, transformed image
    """
    if len(image.shape) != 2:
        raise ValueError('Image must be 2-dimensional')

    flipped_image = image.copy()

    if flip_x:
        for row in range(flipped_image.shape[0]):
            flipped_image[row] = flipped_image[row][::-1]

    if flip_y:
        for column in range(flipped_image.shape[1]):
            flipped_image[:,column] = flipped_image[:,column][::-1]

    return flipped_image

def rotate_image(image, angle):
    """
    Rotates images by a given angle around its center. Points outside of the
    boundaries of the image are filled with value of the nearest point.
    Non-destructive: returns a copy of the input image
    input: image
           angle - rotation angle in degrees in counter-clockwise direction
    output: rotated_image - a new, transformed image
    """
    rotated_image = skimage.transform.rotate(image, angle, mode='nearest')
    return rotated_image
