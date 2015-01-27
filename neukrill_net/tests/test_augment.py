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
        processing = augment.augmentation_wrapper({})
        images2 = image_processing.load_images(self.image_fpaths, processing)
        self.assertListOfNumpyArraysEqual(images, images2)
        
    def test_rotations(self):
        self.assertEqual(len(augment.rotations(self.image, 5)), 5)
        
    def test_augmentation_counts(self):
        """
        Ensure each of the augmentations give the correct number of output images
        """
        # Load up the images to run the test on
        num_images = len(self.image_fpaths)
        images = image_processing.load_images(self.image_fpaths, None)
        self.assertEqual(len(images), num_images)
        
        # Test with resizing
        augment_settings = {'resize':(48,48)}
        processing = augment.augmentation_wrapper(augment_settings)
        procImages = image_processing.load_images(self.image_fpaths, processing)
        self.assertEqual(len(procImages), num_images)
        
        # Test with rotation
        augment_settings = {'rotate':8}
        processing = augment.augmentation_wrapper(augment_settings)
        procImages = image_processing.load_images(self.image_fpaths, processing)
        self.assertEqual(len(procImages), num_images*8)
        
        # Test with reflection
        augment_settings = {'flip':True}
        processing = augment.augmentation_wrapper(augment_settings)
        procImages = image_processing.load_images(self.image_fpaths, processing)
        self.assertEqual(len(procImages), num_images*2)
        # Test without reflection
        augment_settings = {'flip':False}
        processing = augment.augmentation_wrapper(augment_settings)
        procImages = image_processing.load_images(self.image_fpaths, processing)
        self.assertEqual(len(procImages), num_images)
        
        # Test with all
        augment_settings = {'resize':(48,48), 'rotate':5, 'flip':True}
        processing = augment.augmentation_wrapper(augment_settings)
        procImages = image_processing.load_images(self.image_fpaths, processing)
        self.assertEqual(len(procImages), num_images*10)

