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

    DOES NOT WORK YET (MAY NEVER WORK).
    """
    def __init__(self,fn):
        self.fn = fn
        self.cpu_only = False
        self.target_shape = target_shape

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


class SampleAugment(pylearn2.blocks.Block):
    """
    ANOTHER VERSION OF THE ABOVE THAT MAY NEVER WORK.
    """
    def __init__(self,fn,target_shape,input_shape):
        self._fn = fn
        self.cpu_only=False
        self.target_shape = target_shape
        self.input_shape = input_shape
    def __call__(self,inputs):
        return self.fn(inputs)
    def fn(self,inputs):
        # prepare empty array same size as inputs
        req = inputs.shape
        sh = [inputs.shape[0]] + list(self.target_shape)
        inputs = inputs.reshape(sh)
        processed = np.zeros(sh)
        # hand each image as a 2D array
        for i in range(inputs.shape[0]):
            processed[i] = self._fn(inputs[i].reshape(self.target_shape))
        processed = processed.reshape(req)
        processed = processed.astype(np.float32)
        return processed
    def get_input_space(self):
        return self.input_shape
    def get_output_space(self):
        return self.target_shape
