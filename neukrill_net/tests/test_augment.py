#!/usr/bin/env python
"""
Unit tests for image processing functions
"""
import skimage.io
import skimage.transform
import numpy as np
from neukrill_net.tests.base import BaseTestCase
import neukrill_net.image_processing as image_processing
import neukrill_net.augment as augment


class TestAugmentImages(BaseTestCase):
    """
    Unit tests for LoadImage function
    """
    def setUp(self):
        # Grab fpaths from base fpaths dict for test images
        self.image_fpaths = self.image_fname_dict['test']
        # Read the first of the images using skimage
        self.image = skimage.io.imread(self.image_fname_dict['test'][0])

    def test_load_images_without_augmentation(self):
        """
        Test load images returns list of flat images as expected
        with no processing
        """
        # Load the images without a processing function
        images = image_processing.load_images(self.image_fpaths, None)
        # Test the augmentation wrapper can generate a function which does
        # no processing
        processing = augment.augmentation_wrapper(units=None)
        images2 = image_processing.load_images(self.image_fpaths, processing)
        self.assertListOfNumpyArraysEqual(images, images2)
        
    def test_rotations(self):
        # Test numerosity
        self.assertEqual(len(augment.rotations(self.image, 5, True)), 5)
        # Check if we really are using the lossless rotation
        rotatedImages = augment.rotations(self.image, 4, True)
        self.assertEqual(self.image, np.rot90(rotatedImages[-1]))
        
    def test_allcrops(self):
        # Test numerosity
        self.assertEqual(len(augment.allcrops(self.image)), 4)
        
    def test_allpads(self):
        # Test numerosity with no padding
        self.assertEqual(len(augment.allpads(self.image, [0])), 1)
        # Test numerosity with always padded the same absolute amount
        self.assertEqual(len(augment.allpads(self.image, [5])), 4)
        # Test numerosity with 0 or single pad
        self.assertEqual(len(augment.allpads(self.image, [0,5])), 9)
        # Test numerosity with two possible pads
        self.assertEqual(len(augment.allpads(self.image, [2,6])), 16)
        
    def test_augmentation_numerosity(self):
        """
        Ensure each of the augmentations give the correct number of output images
        """
        # Load up the images to run the test on
        num_images = len(self.image_fpaths)
        images = image_processing.load_images(self.image_fpaths, None)
        self.assertEqual(len(images), num_images)
        
        # Test with resizing
        augment_settings = {'resize':(48,48)}
        procImages = image_processing.load_images(self.image_fpaths, 
                        augment.augmentation_wrapper(**augment_settings))
        self.assertEqual(len(procImages), num_images)
        
        # Test with rotation
        augment_settings = {'rotate':8, 'rotate_is_resizable':True}
        procImages = image_processing.load_images(self.image_fpaths, 
                        augment.augmentation_wrapper(**augment_settings))
        self.assertEqual(len(procImages), num_images*8)
        
        # Test with reflection
        augment_settings = {'flip':True}
        procImages = image_processing.load_images(self.image_fpaths, 
                        augment.augmentation_wrapper(**augment_settings))
        self.assertEqual(len(procImages), num_images*2)
        # Test without reflection
        augment_settings = {'flip':False}
        procImages = image_processing.load_images(self.image_fpaths, 
                        augment.augmentation_wrapper(**augment_settings))
        self.assertEqual(len(procImages), num_images)
        
        # Test with crop
        augment_settings = {'crop':True}
        procImages = image_processing.load_images(self.image_fpaths, 
                        augment.augmentation_wrapper(**augment_settings))
        self.assertEqual(len(procImages), num_images*5)
        
        # Test with all so far enabled
        augment_settings = {'resize':(48,48), 'rotate':5, 'rotate_is_resizable':True, 'flip':True, 'crop':True}
        procImages = image_processing.load_images(self.image_fpaths, 
                        augment.augmentation_wrapper(**augment_settings))
        self.assertEqual(len(procImages), num_images*50)
        
        # Test with shape-fixing
        augment_settings = {'shape':(48,48)}
        procImages = image_processing.load_images(self.image_fpaths, 
                        augment.augmentation_wrapper(**augment_settings))
        self.assertEqual(len(procImages), num_images)
        
        # Test with landscapise
        augment_settings = {'landscapise':True}
        procImages = image_processing.load_images(self.image_fpaths, 
                        augment.augmentation_wrapper(**augment_settings))
        self.assertEqual(len(procImages), num_images)
        
        # Test with traslation
        augment_settings = {'translations':[0,5]}
        procImages = image_processing.load_images(self.image_fpaths, 
                        augment.augmentation_wrapper(**augment_settings))
        self.assertEqual(len(procImages), num_images*9)
        
        # Test with noisify
        augment_settings = {'noisify':True}
        procImages = image_processing.load_images(self.image_fpaths, 
                        augment.augmentation_wrapper(**augment_settings))
        self.assertEqual(len(procImages), num_images)
        
        # Test with shear
        augment_settings = {'shear':[0,5,30]}
        procImages = image_processing.load_images(self.image_fpaths, 
                        augment.augmentation_wrapper(**augment_settings))
        self.assertEqual(len(procImages), num_images*3)
        
        
