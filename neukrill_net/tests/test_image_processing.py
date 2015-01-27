#!/usr/bin/env python
"""
Unit tests for image processing functions
"""
import skimage.io
import skimage.transform
import numpy as np
from neukrill_net.tests.base import BaseTestCase
import neukrill_net.image_processing as image_processing

class TestLoadImages(BaseTestCase):
    """
    Unit tests for LoadImage function
    """
    def setUp(self):
        """
        Grab fpaths from base fpaths dict for test images
        """
        self.image_fpaths = self.image_fname_dict['test']

    def test_load_images_without_processing(self):
        """
        Test load images returns list of flat images as expected
        with no processing
        """
        images = image_processing.load_images(self.image_fpaths, None)
        self.assertEqual(len(images), 3)

    def test_load_images_with_min_processing(self):
        """
        Ensure a processing function works
        """
        processing = lambda image: image.min()
        images = image_processing.load_images(self.image_fpaths, processing)
        self.assertEqual(len(images), 3)
        self.assertEqual([[int(x[0])] for x in images], [[63], [5], [46]])

class TestResize(BaseTestCase):
    """
    Unit tests for image resizing as this function is going to expand
    """
    def setUp(self):
        """
        Read the first of the images using skimage
        """
        self.image = skimage.io.imread(self.image_fname_dict['test'][0])

    def test_resize(self):
        """
        Ensure resizing works
        """
        self.assertEqual(
                image_processing.resize_image(self.image, (5,5)).shape, (5,5))
        self.assertEqual(
                image_processing.resize_image(self.image, (2000,2000)).shape,
                         (2000,2000))

class TestFlip(BaseTestCase):
    """
    Unit tests for image flipping
    """
    def setUp(self):
        """
        Read the first of the images using skimage
        """
        self.image = skimage.io.imread(self.image_fname_dict['test'][0])


    def check_images_are_equal(self, image, flipped_image,
                               flip_x=False, flip_y=False):
        """
        Checks whether the values in the two images are equal
        Can check for indices flipped around either axis
        """
        for x in range(image.shape[0]):
            rev_x = x if not flip_y else image.shape[0] - x - 1
            for y in range(image.shape[1]):
                rev_y = y if not flip_x else image.shape[1] - y - 1
                self.assertEqual(image[x, y], flipped_image[rev_x, rev_y])

    def test_flip(self):
        """
        Ensure flipping works
        """
        # Check when flipped in no axes
        flipped_image_x = image_processing.flip_image(self.image)

        self.check_images_are_equal(self.image, flipped_image_x)

        # Check when flipped in X-axis
        flipped_image_x = image_processing.flip_image(self.image, flip_x=True)

        self.check_images_are_equal(self.image, flipped_image_x, flip_x=True)
        
        # Check X flipping is reversible
        self.assertTrue(np.array_equal(
            self.image,
            image_processing.flip_image(flipped_image_x, flip_x=True)))

        # Check when flipped in Y-axis
        flipped_image_y = image_processing.flip_image(self.image, flip_y=True)
        
        self.check_images_are_equal(self.image, flipped_image_y, flip_y=True)
        
        # Check Y flipping is reversible
        self.assertTrue(np.array_equal(
            self.image,
            image_processing.flip_image(flipped_image_y, flip_y=True)))
        
        # Check when flipped in X- & Y-axes
        flipped_image_xy = \
                image_processing.flip_image(self.image,
                                            flip_x=True,
                                            flip_y=True)

        self.check_images_are_equal(self.image, flipped_image_xy, flip_x=True, flip_y=True)
        
        # Check X & Y flipping is reversible
        self.assertTrue(np.array_equal(
            self.image,
            image_processing.flip_image(flipped_image_xy, flip_x=True, flip_y=True)))
        
