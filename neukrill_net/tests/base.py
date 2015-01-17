#!/usr/bin/env python
"""
Objects shared by all the test cases
"""
import os
import glob
import unittest

class BaseTestCase(unittest.TestCase):
    """
    Superclass for all neukrill-net test cases
    """
    @classmethod
    def setUpClass(self):
        self.test_dir = os.path.join('neukrill_net', 'tests', 'resources')

        self.classes = ('acantharia_protist',
                        'acantharia_protist_halo',
                        'artifacts_edge',
                        'fecal_pellet')

        self.image_fname_dict = {'test': sorted(glob.glob(os.path.join(self.test_dir,
                                                       'test', '*.jpg'))),
                        'train': {class_dir: sorted(glob.glob(os.path.join(self.test_dir,
                                                                    'train',
                                                                    class_dir,
                                                                    '*.jpg'))) \
                                                    for class_dir in self.classes}}


