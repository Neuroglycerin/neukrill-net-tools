#!/usr/bin/env python
"""
Unit tests for image processing functions
"""
import skimage.io
import skimage.transform
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
        self.assertEqual(image_processing.resize(self.image, (5,5)).shape, (5,5))
        self.assertEqual(image_processing.resize(self.image, (2000,2000)).shape, (2000,2000))

