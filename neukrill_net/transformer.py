#!/usr/bin/env python
# simple fix for Transformer class that should let us 
# use it to create arbitrary augmentations
# Fix is tracked as an issue but not fixed:
# https://github.com/lisa-lab/pylearn2/issues/1257
# From a thread where someone was trying to do this exact same thing:
# https://groups.google.com/forum/#!searchin/pylearn-users/augmentation/pylearn-users/LLlBpl9z0MY/yQojBR0t2zkJ

import pylearn2.datasets.transformer_dataset

class TransformerDataset(pylearn2.datasets.transformer_dataset.TransformerDataset):
    def __init__(self, raw, transformer, cpu_only=False,
                 space_preserving=False):
        """
            .. todo::

                WRITEME properly

            Parameters
            ----------
            raw : pylearn2 Dataset
                Provides raw data
            transformer: pylearn2 Block
                To transform the data
        """
        self.__dict__.update(locals())
        del self.self
        self.raw.iterator.uneven = False

    @property
    def uneven(self):
        return False
