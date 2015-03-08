#!/usr/bin/env python
"""
Data Augmentation wrappers and functions.
Functions to be used as "processing" arguments
are here.
"""

from __future__ import division

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
        if 'resize_order' not in augment_settings:
            augment_settings['resize_order'] = 0.75
        # apply our resize
        resize = lambda images: [image_processing.resize_image(
                                    image,
                                    augment_settings['resize'],
                                    augment_settings['resize_order'])
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
        # Let 'shapefix' and 'shape' be aliases of the same property
        if 'shapefix' in kwargs:
            kwargs['shape'] = kwargs['shapefix']
        # Store settings as a dictionary
        self.settings = kwargs
        # Add kwargs with defaults into dictionary
        self.settings['units'] = units
        # Don't even need this anymore
        #self.settings['rotate_is_resizable'] = rotate_is_resizable
        
    
    def get_augs(self):
        """
        Computes the augmentation options
        """
        # Flip (always horizontally)
        # All other relfections can be acheived by coupling with an appropriate reflection
        # Flip setting should either be True or False in settings
        if 'flip' in self.settings and self.settings['flip']:
            # Flip if coin-toss says so
            flip = self.rng.binomial(1, 0.5)
        else:
            flip = False
        
        ### Affine Transformation
        # Rescale
        if ('scale' not in self.settings or
                self.settings['scale']==None or
                self.settings['scale']==0.0 ):
            scalefactor = 1.0
        else:
            if isinstance(self.settings['scale'],(list,tuple)):
                sf_index = self.rng.randint(0, len(self.settings['scale']))
                scalefactor = self.settings['scale'][sf_index]
            else:
                scalefactor = self.rng.normal(loc=1.0, scale=self.settings['scale'])
                # Ignore extreme events
                if (scalefactor-1.0) > 3*self.settings['scale']:
                    scalefactor = 1.0
                elif (scalefactor-1.0) < -3*self.settings['scale']:
                    scalefactor = 1.0
        
        # Asymmetric x and y scaling
        if ('scale_asym' not in self.settings
                or self.settings['scale_asym']==None
                or self.settings['scale_asym']==0.0 ):
            # x and y scales are the same
            scalefactor = (scalefactor,scalefactor)
        else:
            # Pick a new x and y scale near the overall scale
            x_sf = self.rng.normal(scale=self.settings['scale_asym'])
            y_sf = self.rng.normal(scale=self.settings['scale_asym'])
            # Ignore extreme events
            if x_sf > 3*self.settings['scale_asym'] or x_sf < -3*self.settings['scale_asym']:
                x_sf = 0.0
            if y_sf > 3*self.settings['scale_asym'] or y_sf < -3*self.settings['scale_asym']:
                y_sf = 0.0
            # Combine x and y scales as a tuple
            scalefactor = (scalefactor+x_sf, scalefactor+y_sf)
        
        # Define translation (shunt)
        if ('shunt' not in self.settings
                or self.settings['shunt']==None
                or self.settings['shunt']==0.0 ):
            shuntlist = (0,0)
        else:
            shuntlist = []
            for i in range(2):
                if isinstance(self.settings['shunt'],(list,tuple)):
                    # Pick from list
                    shunt_index = self.rng.randint(0, len(self.settings['shunt']))
                    shunt = self.settings['shunt'][shunt_index]
                    # Ensure symmetry
                    if self.rng.binomial(1, 0.5):
                        shunt = -shunt
                else:
                    # Chose from gaussian
                    shunt = self.rng.normal(loc=0.0, scale=self.settings['shunt'])
                    # Ignore extreme events
                    if shunt<-3*self.settings['shunt'] or shunt>3*self.settings['shunt']:
                        shunt = 0
                # Add to the list
                shuntlist += [shunt]
            # Convert list to tuple
            shuntlist = tuple(shuntlist)
        
        # Rotate
        if ('rotate' not in self.settings
                or self.settings['rotate']==None
                or self.settings['rotate']==0.0 ):
            rot_angle = 0.0
        else:
            if isinstance(self.settings['rotate'],(list,tuple)):
                # Pick from list of potential rotations
                rot_index = self.rng.randint(0, len(self.settings['rotate']))
                rot_angle = self.settings['rotate'][rot_index]
                
            elif self.settings['rotate']==-1:
                # Uniform distribution of rotations
                rot_angle = self.rng.uniform(low=0.0, high=360.0)
                
            else:
                # Select from implied list
                rot_index = self.rng.randint(0, self.settings['rotate'])
                rot_angle = 360.0 * rot_index / self.settings['rotate']
        
        # Shear
        if ('shear' not in self.settings
                or self.settings['shear']==None
                or self.settings['shear']==0.0 ):
            shear_angle = 0.0
        else:
            if isinstance(self.settings['shear'],(list,tuple)):
                shear_index = self.rng.randint(0, len(self.settings['shear']))
                shear_angle = self.settings['shear'][shear_index]
            else:
                # Chose from gaussian
                shear_angle = self.rng.normal(loc=0.0, scale=self.settings['shear'])
        
        # Order of affine transformation
        if 'transform_order' not in self.settings or self.settings['transform_order']==None:
            transform_order = 0.5
        else:
            transform_order = self.settings['transform_order']
        
        aug_dic = {'flip': flip,
                    'scalefactor': scalefactor, 'rot_angle': rot_angle,
                    'shear_angle': shear_angle, 'shuntlist': shuntlist,
                    'transform_order': transform_order}
        return aug_dic
        
    
    def __call__(self, image):
        """
        Basically a wrapper function for augment_and_process
        """
        
        # Landscapise
        # Set to True if you want to ensure all the images are landscape
        if 'landscapise' in self.settings and self.settings['landscapise']:
            image = image_processing.landscapise_image(image)
        
        # Get the augmentation properties
        aug_dic = self.get_augs()
        
        # Augment and preprocess
        return self.augment_and_process(image, aug_dic, self.settings)
        
        
    def augment_and_process(self, image, aug_dic, processing_settings):
        """
        Maps raw image to augmented image.
        """
        
        # Note down the original type
        original_dtype = image.dtype
        
        # Convert to float while we process it
        image = skimage.util.img_as_float(image)
        
        #####################################################
        # Pre-augmentation processing
        
        # Ensure image is square now if we are going to resize
        if 'resize' in processing_settings:
            image = image_processing.pad_to_square(image)
        
        # Shape-fixing without resizing
        if 'shape' in processing_settings:
            if 'dynamic_shapefix' not in processing_settings or processing_settings['dynamic_shapefix']:
                # Do a dynamic shapefix where we pan to a random location of those viable
                pos_x = self.rng.uniform(low=0.0, high=1.0)
                pos_y = self.rng.uniform(low=0.0, high=1.0)
                image = image_processing.dynamic_shape_fix(image, processing_settings['shape'],
                            (pos_x,pos_y), do_crop=False, do_pad=True)
                
            else:
                image = image_processing.shape_fix(image, processing_settings['shape'],
                            do_crop=False, do_pad=True)
        
        #####################################################
        # Random augmentation
        
        if aug_dic['flip']:
            image = image_processing.flip_image(image, True)
        
        # Perform affine transformation
        image = image_processing.custom_transform_nice_units(image,
                    scale=aug_dic['scalefactor'], rotation=aug_dic['rot_angle'],
                    shear=aug_dic['shear_angle'], translation=aug_dic['shuntlist'],
                    order=aug_dic['transform_order'])
        
        #####################################################
        # Post-augmentation processing
        
        # Shape-fixing without resizing
        if 'shape' in processing_settings:
            if 'dynamic_shapefix' not in processing_settings or processing_settings['dynamic_shapefix']:
                # Do a dynamic shapefix where we pan to a random location of those viable
                image = image_processing.dynamic_shape_fix(image, processing_settings['shape'],
                            (pos_x,pos_y))
                
            else:
                image = image_processing.shape_fix(image, processing_settings['shape'])
        
        # Resize
        if not 'resize_order' in processing_settings or processing_settings['resize_order']==None:
            resize_order = 0.75
        if 'resize' in processing_settings:
            image = image_processing.resize_image(image, processing_settings['resize'], resize_order)
        
        #####################################################
        
        # Add pixel noise
        if ('noise' in self.settings
                and not self.settings['noise']==None
                and not self.settings['noise']==0.0 ):
            noise_seed = self.rng.randint(2**32)
            image = image_processing.noisify_image(image, self.settings['noise'], noise_seed)
        
        #####################################################
        
        if 'normalise' in processing_settings:
            if processing_settings['normalise']['global_or_pixel'] == 'global':
                mu = processing_settings['normalise']['mu']
                sigma = processing_settings['normalise']['sigma']
                image = (image - mu)/sigma
            elif processing_settings['normalise']['global_or_pixel'] == 'pixel':
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        mu = processing_settings['normalise']['mu'][str((i,j))]
                        sigma = processing_settings['normalise']['sigma'][str((i,j))]
                        image[i,j] = (image[i,j] - mu)/sigma
            else:
                raise ValueError("Invalid option for global_or_pixel, should be "
                                     "one of global or pixel.")
        
        #####################################################
        
        # Datatype unit conversion
        if ('units' not in self.settings or
                self.settings['units'] == None or self.settings['units'] == 'auto'):
            # Preserve the datatype
            # Ensure output matches input
            image = image_processing.img_as_dtype(image, original_dtype)
            
        elif self.settings['units'] == 'float64' or self.settings['units'] == 'float':
            image = skimage.util.img_as_float(image)
            
        elif self.settings['units'] == 'uint8':
            image = skimage.util.img_as_ubyte(image)
            
        else:
            raise ValueError('Unrecognised output units: {}'.format(
                                self.settings['units']))
        
        return image
        
        
class ParallelRandomAugment(RandomAugment):
    """
    Random augmentation, but more than one at a time!
    """
    def __init__(self, preproc_list, **kwargs):
        
        # Assign preprocessing options to attributes
        self.preproc_list = preproc_list
        
        # Call superclass
        RandomAugment.__init__(self, **kwargs)
        
    def __call__(self, image):
        """
        Wraps two augment_and_process functions and returns
        two results as a tuple
        """
        
        # Landscapise
        # Set to True if you want to ensure all the images are landscape
        if 'landscapise' in self.settings and self.settings['landscapise']:
            image = image_processing.landscapise_image(image)
        
        # Get the augmentation properties
        aug_dic = self.get_augs()
        
        # Augment and preprocess
        images = []
        for preproc in self.preproc_list:
            images += [self.augment_and_process(image, aug_dic, preproc)]
        
        return images
        
