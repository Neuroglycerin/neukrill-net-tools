#!/usr/bin/env python
"""
Unit tests for utility functions
"""
import os
import glob
import shutil
import io
from neukrill_net.tests.base import BaseTestCase
import neukrill_net.utils as utils

class TestSettingsParser(BaseTestCase):
    """
    Unit tests for utils.parse_settings
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

        self.check_settings_string = io.StringIO(
                                            '{{"data_dir": ["{0}"]}}'.format(self.data_dir))

        self.check_string_parse_is_same_as_file(self.check_settings_file,
                                                self.check_settings_string)


    def check_string_parse_is_same_as_file(self,
                                           string_settings,
                                           file_settings):
        """
        Ensure that stringIO objects used in testing return the same as a
        real test file
        """

        self.assertEqual(utils.parse_settings(string_settings),
                         utils.parse_settings(file_settings))


    def test_parse_file_returns_dict(self):
        """
        Ensure a valid dict is returned by function when given a file
        """
        settings = utils.parse_settings(self.check_settings_file)
        self.assertIs(settings.__class__, dict)
        self.assertTrue(len(settings) > 0)

    def test_error_if_file_does_not_exist(self):
        """
        Ensure an IOError is thrown if the file doesn't exist
        """
        with self.assertRaises(IOError):
            utils.parse_settings('fake_file')

    def test_error_if_required_missing(self):
        """
        Ensure error is thrown if a required setting is omitted
        """
        setting_string_without_required = io.StringIO('{"foo": 5, "bar": "duck"}')
        with self.assertRaises(ValueError):
            utils.parse_settings(setting_string_without_required)

    def test_resolves_to_correct_dir(self):
        """
        Make sure settings parser resolves to dir containing test and train dirs
        """
        settings_string_with_2_dirs = io.StringIO('{"data_dir": ["fake", "TestSettingsParserDir"]}')
        settings = utils.parse_settings(settings_string_with_2_dirs)
        self.assertEqual(settings['data_dir'], os.path.abspath(self.data_dir))

    def tearDown(self):
        """
        Remove data dir
        """
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)

if __name__ == '__main__':
    unittest.main()
