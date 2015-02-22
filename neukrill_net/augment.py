#!/usr/bin/env python
"""
Data Augmentation wrappers and functions.
Functions to be used as "processing" arguments
are here.
"""

import neukrill_net.image_processing as image_processing
import numpy as np
import itertools

def augmentation_wrapper(augment_settings):
    """
    Takes settings for augmentation as a dictionary
    and produces a "processing" function to make more
    training data.
    The returned function will return a list of images
    containing EVERY possible combinations of options
    as given in the augmentation settings.
    """
    # components of the processing pipeline
    components = []
    
    # Shape-fixing without resizing
    if 'shape' in augment_settings:
        # 
        shapefix = lambda images: [image_processing.shape_fix(
                                    image,
                                    augment_settings['shape'])
                                    for image in images]
    else:
        shapefix = lambda images: images
        
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
                                    rotations(image,
                                        augment_settings['rotate'],
                                        augment_settings['rotate_is_resizable']
                                    )
                                ]
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
    
    # Crop (every side or not at all)
    if 'crop' in augment_settings and augment_settings['crop']:
        crop = lambda images: images + [croppedImage for image in images
                                            for croppedImage in
                                            allcrops(image)]
    else:
        crop = lambda images: images
    
    # Pad to translate within shape_fix window
    if 'traslations' in augment_settings:
        pad = lambda images: [paddedImage for image in images
                                            for paddedImage in
                                            allpads(image, augment_settings['traslations'])]
    else:
        pad = lambda images: images
    
    # Stack all our functions together
    # Order matters here:
    # - Rotate first because then it has higher accuracy
    #   (might want to move in pixels which would otherwise be cropped
    #    don't want the loss of resolution caused by resize)
    # - Crop
    # - Flip
    # - Resize last because it is lossy
    processing = lambda image: resize( shapefix( pad( crop( flip( rotate( [image] ))))))
    
    return processing

def rotations(image, num_rotations, resizable):
    """
    Returns a number of rotations depending on the settings.
    """
    return [image_processing.rotate_image(image, angle, resizable)
        for angle in np.linspace(0, 360, num_rotations, endpoint=False)]
        
def allcrops(image):
    """
    Returns a number of cropped copies of an image
    """
    return [image_processing.crop_image(image,side_id)
        for side_id in range(4)]

def allpads(image, pad_amounts):
    """
    Returns a list of padded images, with centre shift amounts 
    specified by pad_amounts
    """
    # Can go in all four directions, so include +ve and -ve
    pad_amounts = np.array(pad_amounts)
    pad_amounts = np.union1d(-pad_amounts, pad_amounts)
    
    # Make permutations of all x and y shifts
    return [image_processing.padshift_image(image, shift)
        for shift in itertools.permutations(pad_amounts, 2)]
    
