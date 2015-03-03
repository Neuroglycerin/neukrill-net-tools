#!/usr/bin/env python
# simple fix for Transformer class that should let us 
# use it to create arbitrary augmentations
# Fix is tracked as an issue but not fixed:
# https://github.com/lisa-lab/pylearn2/issues/1257
# From a thread where someone was trying to do this exact same thing:
# https://groups.google.com/forum/#!searchin/pylearn-users/augmentation/pylearn-users/LLlBpl9z0MY/yQojBR0t2zkJ

import pylearn2.datasets.transformer

class TransformerDataset(pylearn2.datasets.transformer_dataset.TransformerDataset):
    @property
    def uneven(self):
        return self.raw_iterator.uneven
