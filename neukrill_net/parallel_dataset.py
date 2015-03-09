"""
Dataset class that wraps the dataset class found in 
image_directory_dataset to support models with branched
input layers; allowing different versions of images as
input to those layers.

Based on dev work in the Interactive Pylearn2 notebook.
"""
__authors__ = "Gavin Gray"
__copyright__ = "Copyright 2015 - University of Edinburgh"
__credits__ = ["Gavin Gray"]
__license__ = "MIT"
__maintainer__ = "Gavin Gray"
__email__ = "gavingray1729@gmail.com"


import numpy as np

import neukrill_net.image_directory_dataset

class ParallelIterator(neukrill_net.image_directory_dataset.FlyIterator):
    """
    A simple version of FlyIterator that is able to deal with multiple
    images being returned by the processing function.
    """
    def next(self):
        # return one batch
        if len(self.indices) >= self.batch_size:
            batch_indices = [self.indices.pop(0) for i in range(self.batch_size)]
            # preallocate array
            if len(self.final_shape) == 2: 
                Xbatch1 = np.zeros([self.batch_size]+list(self.final_shape)+[1])
                Xbatch2 = np.zeros([self.batch_size]+list(self.final_shape)+[1])
            elif len(self.final_shape) == 3:
                Xbatch1 = np.zeros([self.batch_size]+list(self.final_shape))
                Xbatch2 = np.zeros([self.batch_size]+list(self.final_shape))
            # iterate over indices, applying the dataset's processing function
            for i,j in enumerate(batch_indices):
                import pdb
                pdb.set_trace()
                Xbatch1[i],Xbatch2[i] = [image.reshape(Xbatch1.shape[1:]) for 
                        image in self.dataset.fn(self.dataset.X[j])]
            # get the batch for y as well
            ybatch = self.dataset.y[batch_indices,:].astype(np.float32)
            Xbatch1 = Xbatch1.astype(np.float32)
            Xbatch2 = Xbatch2.astype(np.float32)
            return Xbatch1,Xbatch2,ybatch
        else:
            raise StopIteration

class ParallelDataset(neukrill_net.image_directory_dataset.ListDataset):
    """
    Only difference here is that it will use the above Parallel
    iterator instead of the existing iterator.
    """
    def iterator(self, mode=None, batch_size=None, num_batches=None, rng=None,
                        data_specs=None, return_tuple=False):
        """
        Returns iterator object with standard Pythonic interface; iterates
        over the dataset over batches, popping off batches from a shuffled 
        list of indices.
        Inputs:
            - mode: 'sequential' or 'shuffled_sequential'.
            - batch_size: required, size of the minibatches produced.
            - num_batches: supply if you want, the dataset will make as many
        as it can if you don't.
            - rng: not used, as above.
            - data_specs: not used, as above
            - return_tuple: not used, as above
        Outputs:
            - instance of FlyIterator, see above.
        """
        if not num_batches:
            # guess that we want to use all of them
            num_batches = int(len(self.X)/batch_size)
        iterator = ParallelIterator(dataset=self, batch_size=batch_size, 
                                num_batches=num_batches, 
                                final_shape=self.run_settings["final_shape"],
                                rng=self.rng)
        return iterator
