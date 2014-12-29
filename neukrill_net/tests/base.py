#!/usr/bin/env python
"""
Objects shared by all the test cases
"""
import os
import unittest

class BaseTestCase(unittest.TestCase):
    """
    Superclass for all neukrill-net test cases
    """
    test_dir = os.path.join('neukrill_net', 'tests', 'resources')
