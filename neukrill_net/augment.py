#!/usr/bin/env python
"""
Data Augmentation wrappers and functions.
Functions to be used as "processing" arguments
are here.
"""

import neukrill_net.image_processing as image_processing
import numpy as np
import skimage.util
import itertools

def augmentation_wrapper(units='float64', **augment_settings):
    """
    Takes settings for augmentation as **kwargs
    and produces a "processing" function to make more
    training data.
    The returned function will return a list of images
    containing EVERY possible combinations of options
    as given in the augmentation settings.
    """
    # components of the processing pipeline
    components = []
    
    # Datatype unit conversion
    # We very probably want this to be float throughout
    if units == 'float64' or units == 'float':
        unitconvert = lambda images: [skimage.util.img_as_float(image)
                                        for image in images]
        
    elif units == 'uint8':
        unitconvert = lambda images: [skimage.util.img_as_ubyte(image)
                                        for image in images]
    
    elif units == None or units == 'auto':
        unitconvert = lambda images: images
    
    else:
        raise ValueError('Unrecognised output units: {}'.format(
                            augment_settings['units']))
    
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
    
    
    # Rescale
    if 'scale' in augment_settings and not augment_settings['scale']==None:
        scale = lambda images: [image_processing.scale_image(image, sf)
                                for sf in augment_settings['scale']
                                for image in images]
    else:
        scale = lambda images: images
    
    
    # Landscapise
    # Set to True if you want to ensure all the images are landscape
    if 'landscapise' in augment_settings and augment_settings['landscapise']:
        landscapise = lambda images: [image_processing.landscapise_image(image)
                                        for image in images]
    else:
        landscapise = lambda images: images
    
    
    # Rotate
    if 'rotate' in augment_settings and not augment_settings['rotate']==None:
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
    
    # Shear
    if 'shear' in augment_settings and not augment_settings['shear']==None:
        shear = lambda images: [image_processing.shear_image(image, this_shear)
                                for this_shear in augment_settings['shear']
                                for image in images]
    else:
        shear = lambda images: images
    
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
    if 'translations' in augment_settings and not augment_settings['translations']==None:
        pad = lambda images: [paddedImage for image in images
                                            for paddedImage in
                                            allpads(image, augment_settings['translations'])]
    else:
        pad = lambda images: images
    
    
    # Add pixel noise
    if 'noise' in augment_settings and not augment_settings['noise']==None:
        noisify = lambda images: [image_processing.noisify_image(image, augment_settings['noise'])
                                    for image in images]
    else:
        noisify = lambda images: images
    
    
    # Stack all our functions together
    # Order matters here:
    # - Landscapise first because we don't want to undo rotations
    # - Rotate before anything else because then it has higher accuracy
    #   (might want to move in pixels which would otherwise be cropped
    #    don't want the loss of resolution caused by resize)
    # - Flip
    # - Crop
    # - Pad
    # - Align shape window
    # - Resize last because it is lossy
    # - Add noise independent for all output pixels
    processing=lambda image:noisify(
                                resize(
                                    shapefix(
                                        pad(
                                            crop(
                                                shear(
                                                    flip(
                                                        rotate(
                                                            landscapise(
                                                                unitconvert(
                                                                    [ image ]
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
    
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
        for shift in itertools.product(pad_amounts, repeat=2)]


class RandomAugment(object):
    """
    A class for performing random augmentations.
    When you call it with an image it gives
    you back a randomly manipulated version of it.
    """
    
    def __init__(self, random_seed=42, units='float64', rotate_is_resizable=False, **kwargs):
        """
        Inputs: random_seed
                keyword assignment of augmentation settings.
        """
        # Initialise the random number generator
        self.rng = np.random.RandomState(seed=random_seed)
        # Store settings as a dictionary
        self.settings = kwargs
        # Add kwargs with defaults into dictionary
        self.settings['units'] = units
        self.settings['rotate_is_resizable'] = rotate_is_resizable
        
    
    def __call__(self, image):
        """
        Maps raw image to augmented image.
        """
        
        # Datatype unit conversion
        # We very probably want this to be float throughout
        if 'units' not in self.settings or self.settings['units'] == None or self.settings['units'] == 'auto':
            pass
        elif self.settings['units'] == 'float64' or self.settings['units'] == 'float':
            image = skimage.util.img_as_float(image)
        elif self.settings['units'] == 'uint8':
            image = skimage.util.img_as_ubyte(image)
        else:
            raise ValueError('Unrecognised output units: {}'.format(
                                self.settings['units']))
        
        # Note down the original type
        original_dtype = image.dtype
        
        #####################################################
        # Pre-augmentation processing
        
        # Landscapise
        # Set to True if you want to ensure all the images are landscape
        if 'landscapise' in self.settings and self.settings['landscapise']:
            image = image_processing.landscapise_image(image)
        
        #####################################################
        # Random augmentation
        
        # Rescale
        if 'scale' in self.settings and not self.settings['scale']==None:
            sf_index = self.rng.randint(0, len(self.settings['scale']))
            image = image_processing.scale_image(image, self.settings['scale'][sf_index])
        
        # Define shunts
        if 'shunt' in self.settings and not self.settings['shunt']==None:
            shuntlist = []
            for i in range(4):
                if type(self.settings['shunt']) is list:
                    # Pick from list
                    shunt_index = self.rng.randint(0, len(self.settings['shunt']))
                    shunt = self.settings['shunt'][shunt_index]
                else:
                    # Chose from gaussian
                    shunt = self.rng.normal(loc=0.0, scale=self.settings['shunt'])
                    # Use some hard caps, for very unlikely events
                    if shunt<-0.4:
                        shunt = -0.4
                    if shunt>0.4:
                        shunt = 0.4
                # Add to the list
                shuntlist += [shunt]
        
        # Perform shunts if they are pads now
        if 'shunt' in self.settings and not self.settings['shunt']==None:
            for i in range(4):
                if shuntlist[i]>0:
                    image = image_processing.padcrop_image(image, i, shuntlist[i])
        
        # Rotate
        if 'rotate' in self.settings and not self.settings['rotate']==None:
            if type(self.settings['rotate']) is list:
                # Pick from list of potential rotations
                rot_index = self.rng.randint(0, len(self.settings['rotate']))
                rot_angle = self.settings['rotate'][rot_index]
                
            elif self.settings['rotate']==-1:
                # Uniform distribution of rotations
                rot_angle = self.rng.uniform(low=0.0, high=360.0)
                
            else:
                # Select from implied list
                rot_index = self.rng.randint(0, self.settings['rotate'])
                rot_angle = 360 * rot_index / rot_angle
                
            image = image_processing.rotate_image(
                        image, rot_angle,
                        resizable=self.settings['rotate_is_resizable'])
        
        # Shear
        if 'shear' in self.settings and not self.settings['shear']==None:
            shear_index = self.rng.randint(0, len(self.settings['shear']))
            image = image_processing.shear_image(image, shear_index)
        
        # Flip (always horizontally)
        # All other relfections can be acheived by coupling with an appropriate reflection
        # Flip setting should either be True or False in settings
        if 'flip' in self.settings and self.settings['flip']:
            # Flip if coin-toss says so
            if self.rng.binomial(1, 0.5):
                image = image_processing.flip_image(image, True)
        
        # Crop (every side or not at all)
        if 'crop' in self.settings and not self.settings['crop']==None:
            for side_id in range(4):
                crop_index = self.rng.randint(0, len(self.settings['crop']))
                image = image_processing.crop_image(image, side_id, 
                        self.settings['crop'][crop_index])
        
        # Perform shunts if they are crops now
        if 'shunt' in self.settings and not self.settings['shunt']==None:
            for i in range(4):
                if shuntlist[i]<0:
                    image = image_processing.padcrop_image(image, i, shuntlist[i])
        
        #####################################################
        # Post-augmentation processing
        
        # Shape-fixing without resizing
        if 'shape' in self.settings:
            image = image_processing.shape_fix(image, self.settings['shape'])
        
        # Resize
        if 'resize' in self.settings:
            image = image_processing.resize_image(image, self.settings['resize'])
        
        #####################################################
        
        # Add pixel noise
        if 'noise' in self.settings and not self.settings['noise']==None:
            noise_seed = self.rng.randint(2**32)
            image = image_processing.noisify_image(image, self.settings['noise'], noise_seed)
        
        #####################################################
        # Preserve the datatype
        # Ensure output matches input
        image = image_processing.img_as_dtype(image, original_dtype)
        
        return image
        
        
