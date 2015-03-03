#!/usr/bin/env python
"""
Class that inherits from the Pylearn2 Block class (although that class does
not appear to do very much) and defines a transformation to be performed on
batches of images in a Transformer Dataset.
"""

import pylearn2.blocks
import numpy as np

class AugmentBlock(pylearn2.blocks.Block):
    """
    This Block takes any function object that can process
    images and applies it to all the images in a batch it's
    given to process.
    """
    def __init__(self,fn):
        self.fn = fn
        self.cpu_only = False

    def __call__(self,inputs):
        """
        Expect to be given a batch of images as a 4D numpy array. 
        Process images and return a new array with the same 
        first dimension (expected to be number of examples).
        """
        # prepare array
        processed = np.zeros(inputs.shape)
        # hand each image as a 2D array
        for i in range(inputs.shape[0]):
            processed[i] = self.fn(inputs[i].reshape(
                        inputs.shape[1:3]))[np.newaxis].T
        return processed

