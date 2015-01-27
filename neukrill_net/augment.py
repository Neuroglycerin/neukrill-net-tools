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
    
    # Resize
    if 'resize' in augment_settings:
        # apply our resize
        resize = lambda images: [image_processing.resize_image(
                                    image,
                                    augment_settings['resize'])
                                    for image in images]
    else:
        resize = lambda images: images
    
    # Rotate
    if 'rotate' in augment_settings:
        # apply rotate with options
        # will require a function here to call it a number of times
        rotate = lambda images: [rotatedImage for image in images
                                    for rotatedImage in
                                    rotations(image, augment_settings['rotate'])]
    else:
        rotate = lambda images: images
    
    # Flip (always horizontally)
    # All other relfections can be acheived by coupling with an appropriate reflection
    # Flip setting should either be True or False in settings
    if 'flip' in augment_settings and augment_settings['flip']:
        flip = lambda images: images + [image_processing.flip_image(image, True)
                                            for image in images]
    else:
        flip = lambda images: images
    
    # Stack all our functions together
    # Order matters here:
    # - Rotate first because then it has higher accuracy
    #   (might want to move in pixels which would otherwise be cropped
    #    don't want the loss of resolution caused by resize)
    # - Crop
    # - Flip
    # - Resize last because it is lossy
    processing = lambda image: resize(flip(rotate([image])))
    
    return processing

def rotations(image, num_rotations):
    """
    Returns a number of rotations depending on the settings.
    """
    return [image_processing.rotate_image(image,angle) 
        for angle in np.linspace(0,360,num_rotations)]

