#!/usr/bin/env python
"""
Image attributes
Find simple attributes of images.
All these functions return a scalar.
"""

from __future__ import division

import numpy as np
import skimage.util

def height(image):
    """The height of the image"""
    return image.shape[0]
    
def width(image):
    """The width of the image"""
    return image.shape[1]
    
def numpixels(image):
    """The number of pixels in the image"""
    return image.size
    
def aspectratio(image):
    """The aspect ratio of the image"""
    return image.shape[1] / image.shape[0]
    
def mean(image):
    """The mean pixel value"""
    return image.mean()
    
def median(image):
    """The median pixel value"""
    return np.median(image)
    
def var(image):
    """The variance of pixel intensities"""
    return image.var()
    
def std(image):
    """The standard deviation of pixel intensities"""
    return image.std()
    
def stderr(image):
    """The standard error of pixel intensities.
    (Which is standard deviation, normalised by
    square root of number of pixels.)"""
    return image.std() / np.sqrt(image.size)
    
def numwhite(image):
    """The number of pixels in the image which are
    entirely white."""
    # Input is typically uint8
    # Ensure it actually is
    image = skimage.util.img_as_ubyte(image)
    return np.count_nonzero(image == 255)
    
def propwhite(image):
    """The proportion of pixels in the image which are
    entirely white."""
    return numwhite(image) / numpixels(image)
    
def numnonwhite(image):
    """The number of pixels in the image which aren't
    100% white."""
    # Input is typically uint8
    # Ensure it actually is
    image = skimage.util.img_as_ubyte(image)
    return np.count_nonzero(image < 255)
    
def propnonwhite(image):
    """The proportion of pixels in the image which aren't
    100% white."""
    return numnonwhite(image) / numpixels(image)
    
def numblack(image):
    """The number of pixels in the image which are
    entirely black."""
    # Input is typically uint8
    # Ensure it actually is
    image = skimage.util.img_as_ubyte(image)
    return np.count_nonzero(image == 0)
    
def propblack(image):
    """The proportion of pixels in the image which are
    entirely black."""
    return numblack(image) / numpixels(image)
    
def numbool(image):
    """The number of pixels in the image which are more than
    half intensity."""
    # Input is typically uint8
    # Ensure it actually is
    image = skimage.util.img_as_ubyte(image)
    return np.count_nonzero(image > 127)
    
def propbool(image):
    """The proportion of pixels in the image which are more than
    half intensity."""
    return numbool(image) / numpixels(image)
