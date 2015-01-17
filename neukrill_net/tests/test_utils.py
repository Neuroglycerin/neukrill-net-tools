#!/usr/bin/env python
"""
Unit tests for utility functions
"""
import os
import shutil
import io
import numpy as np
from neukrill_net.tests.base import BaseTestCase
import neukrill_net.utils as utils
import neukrill_net.utils as constants
import unittest.mock

class TestSettings(BaseTestCase):
    """
    Unit tests for settings class
    """

    def setUp(self):
        """
        Set up tests by ensuring that the stringIO object used in testing
        returns the same output as a real json file
        """
        self.data_dir = "TestSettingsParserDir"
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
            os.mkdir(os.path.join(self.data_dir, 'test'))
            os.mkdir(os.path.join(self.data_dir, 'train'))

        self.check_settings_file = os.path.join(self.test_dir,
                                                'check_settings_file.json')

    def test_init_with_correct_file_input(self):
        """
        Ensure a valid dict is created from a correct json during init
        """
        settings = utils.Settings(self.check_settings_file)
        self.assertIs(settings.user_input.__class__, dict)
        self.assertTrue(len(settings.user_input) > 0)

    def test_init_with_correct_stringIO(self):
        """
        Ensure a valid dict is created from a correct json io.String during init
        """
        string_settings = io.StringIO('{"data_dir": ["TestSettingsParserDir"]}')
        settings = utils.Settings(string_settings)
        self.assertIs(settings.user_input.__class__, dict)
        self.assertTrue(len(settings.user_input) > 0)

    def test_error_if_file_does_not_exist(self):
        """
        Ensure an IOError is thrown if the file doesn't exist
        """
        with self.assertRaises(ValueError):
            utils.Settings('fake_file')

    def test_error_if_required_missing(self):
        """
        Ensure error is thrown if a required setting is omitted
        """
        setting_string_without_required = io.StringIO('{"foo": 5, "bar": "duck"}')
        with self.assertRaises(ValueError):
            utils.Settings(setting_string_without_required)

    def test_resolves_to_correct_dir(self):
        """
        Make sure settings parser resolves to dir containing test and train dirs
        """
        settings_string_with_2_dirs = io.StringIO('{"data_dir": ["fake", "TestSettingsParserDir"]}')
        settings = utils.Settings(settings_string_with_2_dirs)
        self.assertEqual(settings.data_dir, os.path.abspath(self.data_dir))

    def check_default_values_during_init(self):
        """
        Make sure default values are set for r_seed
        """
        settings = utils.Settings(self.check_settings_file)
        classes = constants.classes

        self.assertIs(settings.random_seed, 42)
        self.assertIs(settings.classes, classes)

    def tearDown(self):
        """
        Remove data dir
        """
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)


class TestLoadData(BaseTestCase):
    """
    Test load data util function
    """

    def setUp(self):
        """
        Initialise image_fname_dict and classes
        """
        # unecessary but just to make it clear these values are coming from
        # superclass
        self.image_fname_dict = self.image_fname_dict
        self.classes = self.classes

        self.processing = lambda image: np.zeros((10,10))

    def test_load_train_without_processing(self):
        """
        Check load_data fails stacking training data without a processing step
        """
        with self.assertRaises(ValueError):
            _, _ = utils.load_data(self.image_fname_dict,
                                   classes=self.classes)

    def test_loading_train_data_with_processing(self):
        """
        Ensure load_data with training data returns the correct data
        """
        data, labels = utils.load_data(self.image_fname_dict,
                                       classes=self.classes,
                                       processing=self.processing)

        self.assertIs(len(labels), 10)
        self.assertEqual(['acantharia_protist'] * 3 + \
                         ['acantharia_protist_halo'] * 2 + \
                         ['artifacts_edge'] * 4 + \
                         ['fecal_pellet'], list(labels))
        self.assertEqual(data.shape, (10, 100))

    def test_load_train_data_name_correspondence_is_correct(self):
        """
        Ensure the correspondence of labels to data is maintained
        on load
        """
        single_val_processing = lambda images: images.min()

        data, labels = utils.load_data(self.image_fname_dict,
                                       classes=self.classes,
                                       processing=single_val_processing)

        self.assertIs(len(labels), 10)
        self.assertEqual(['acantharia_protist'] * 3 + \
                         ['acantharia_protist_halo'] * 2 + \
                         ['artifacts_edge'] * 4 + \
                         ['fecal_pellet'], list(labels))
        self.assertEqual(data.shape, (10, 1))
        self.assertEqual([[int(x[0])] for x in data], [[51], [73], [65], [35],
                                                       [37], [202], [0], [0],
                                                       [0], [158]])
    def test_load_test_fails_without_processing(self):
        """
        Make sure load_data fails to stack training data without processing
        """
        with self.assertRaises(ValueError):
            _, _ = utils.load_data(self.image_fname_dict)

    def test_loading_test_data_with_processing(self):
        """
        Check whether data and names are correct when loading test data
        with dummy zeros((10,10)) processing
        """
        data, names = utils.load_data(self.image_fname_dict,
                                      processing=self.processing)

        self.assertEqual(names, ['136177.jpg', '81949.jpg', '27712.jpg'])
        self.assertEqual(data.shape, (3, 100))

    def test_load_test_data_name_correspondence_is_correct(self):
        """
        Make sure the names match up to the correct row in the data for test
        data
        """
        single_val_processing = lambda images: images.min()
        data, names = utils.load_data(self.image_fname_dict,
                                      processing=single_val_processing)

        self.assertEqual(names, ['136177.jpg', '81949.jpg', '27712.jpg'])
        self.assertIs(int(data[0][0]), 63)
        self.assertIs(int(data[1][0]), 46)
        self.assertIs(int(data[2][0]), 5)

