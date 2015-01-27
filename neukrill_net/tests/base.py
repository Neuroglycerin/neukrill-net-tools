#!/usr/bin/env python
"""
Objects shared by all the test cases
"""
import os
import glob
import unittest
import numpy as np

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

    
    def __init__(self, *args, **kw):
        """Add test for numpy type"""
        super().__init__(*args, **kw)
        self.addTypeEqualityFunc(np.ndarray, self.assertNumpyEqual)
        
        
    def IsNumpy(self, x):
        return type(x).__module__ == np.__name__
    
    
    def assertNumpyEqual(self, x, y, msg=None):
        if not self.IsNumpy(x) or not self.IsNumpy(y):
            self.failureException("This isn't a numpy array." + msg)
        if x.shape != y.shape:
            self.failureException("Shapes don't match." + msg)
        if not np.allclose(x, y):
            self.failureException("Elements don't match." + msg)
    
    def assertListOfNumpyArraysEqual(self, x, y):
        if len(x) != len(y):
            raise AssertionError("Number of elements don't match")
        for index in range(len(x)):
            if type(x[index]) != type(y[index]):
                raise AssertionError("Class types don't match")
            if self.IsNumpy(type(x[index])):
                self.assertNumpyEqual(self, x[index], y[index])
            else:
                self.assertEqual(x[index], y[index])
    
