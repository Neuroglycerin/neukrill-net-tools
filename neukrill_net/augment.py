#!/usr/bin/env python
"""
Data Augmentation wrappers and functions.
Functions to be used as "processing" arguments
are here.
"""

import neukrill_net.image_processing as image_processing
import numpy as np

def augmentation_wrapper(augment_settings):
    """
    Takes settings for augmentation as a dictionary
    and produces a "processing" function to  
    """
    # components of the processing pipeline
    components = []
    # at the moment can only do resize and rotate
    if "resize" in augment_settings:
        # apply our resize
        resize = lambda image: image_processing.resize_image(image,
                                        augment_settings['resize'])
    else:
        resize = lambda image: image

    if "rotate" in augment_settings:
        # apply rotate with options
        # will require a function here to call it a number of times
        rotate = lambda image: rotations(image, augment_settings)
    else:
        rotate = lambda image: image

    if "flip" in augment_settings:
        flip = lambda image: image_processing.flip_image(image, 
                                            augment_settings['flip'])
        processing = lambda image: rotate(resize(image)) + [flip(resize(image))]
    else:
        processing = lambda image: rotate(resize(image))

    return processing

def rotations(image, augment_settings):
    """
    Returns a number of rotations depending on the settings.
    """
    rotations = lambda image: [image_processing.rotate_image(image,angle) 
        for angle in np.linspace(0,360,augment_settings['rotate'])]
    return rotations(image)

