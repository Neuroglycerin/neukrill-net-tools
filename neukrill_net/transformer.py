#!/usr/bin/env python
# simple fix for Transformer class that should let us 
# use it to create arbitrary augmentations
# Fix is tracked as an issue but not fixed:
# https://github.com/lisa-lab/pylearn2/issues/1257
# From a thread where someone was trying to do this exact same thing:
# https://groups.google.com/forum/#!searchin/pylearn-users/augmentation/pylearn-users/LLlBpl9z0MY/yQojBR0t2zkJ

import pylearn2.datasets.transformer_dataset

class TransformerDataset(pylearn2.datasets.transformer_dataset.TransformerDataset):
    def __init__(self, raw_iterator, transformer_dataset, data_specs):
        """
        .. todo::

            WRITEME
        """
        self.raw_iterator = raw_iterator
        self.transformer_dataset = transformer_dataset
        self.stochastic = raw_iterator.stochastic
        self.data_specs = data_specs

    @property
    def uneven(self):
        return self.raw_iterator.uneven
