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
        # super(self).__init__(*args, **kw) # Only works on Python3
        super(BaseTestCase, self).__init__(*args, **kw) # Works on Python2
        self.addTypeEqualityFunc(np.ndarray, self.assertNumpyEqual)
        
        
    def IsNumpy(self, x):
        return type(x).__module__ == np.__name__
    
    
    def assertNumpyEqual(self, x, y, msg=None):
        if not self.IsNumpy(x) or not self.IsNumpy(y):
            # This is how you are supposed to do it
            # but does not work for me in Python 2.7
            self.failureException("This isn't a numpy array. %s" % msg)
            # This always works
            self.fail("This isn't a numpy array. %s" % msg)
        if not x.shape == y.shape:
            # This is how you are supposed to do it
            # but does not work for me in Python 2.7
            self.failureException("Shapes don't match. %s" % msg)
            # This always works
            self.fail("Shapes don't match. %s" % msg)
        if not np.allclose(x, y):
            # This is how you are supposed to do it
            # but does not work for me in Python 2.7
            self.failureException("Elements don't match. %s" % msg)
            # This always works
            self.fail("Elements don't match. %s" % msg)
    
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
    
