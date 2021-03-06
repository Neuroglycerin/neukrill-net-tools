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
import sklearn.externals

import neukrill_net.image_directory_dataset
import neukrill_net.utils

class ParallelIterator(neukrill_net.image_directory_dataset.FlyIterator):
    """
    A simple version of FlyIterator that is able to deal with multiple
    images being returned by the processing function.
    """
    def next(self):
        # check if we reached the end yet
        if self.final_iteration:
            raise StopIteration

        # allocate array
        if len(self.final_shape) == 2: 
            Xbatch1,Xbatch2 = [np.array(batch).reshape(
                self.batch_size, self.final_shape[0], self.final_shape[1], 1) 
            for batch in zip(*self.result.get(timeout=10.0))]
        elif len(self.final_shape) == 3:
            Xbatch1,Xbatch2 = [np.array(batch)
            for batch in zip(*self.result.get(timeout=10.0))]
        # make sure they're float32
        Xbatch1 = Xbatch1.astype(np.float32)
        Xbatch2 = Xbatch2.astype(np.float32)

        # get y if we're training
        if self.train_or_predict == "train":
            ybatch = self.dataset.y[self.batch_indices,:].astype(np.float32)

        # start processing next batch
        if len(self.indices) >= self.batch_size:
            self.batch_indices = [self.indices.pop(0) 
                                for i in range(self.batch_size)]
            self.result = self.dataset.pool.map_async(self.dataset.fn,
                        [self.dataset.X[i] for i in self.batch_indices])
        else:
            self.final_iteration += 1

        # if training return X and y, otherwise
        # we're testing so return just X
        if self.train_or_predict == "train":
            return Xbatch1,Xbatch2,ybatch
        elif self.train_or_predict == "test":
            return Xbatch1,Xbatch2
        else:
            raise ValueError("Invalid option for train_or_predict:"
                    " {0}".format(self.train_or_predict))

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
            - instance of ParallelIterator, see above.
        """
        if not num_batches:
            # guess that we want to use all of them
            num_batches = int(len(self.X)/batch_size)
        iterator = ParallelIterator(dataset=self, batch_size=batch_size, 
                                num_batches=num_batches, 
                                final_shape=self.run_settings["final_shape"],
                                rng=self.rng, mode=mode, 
                                train_or_predict=self.train_or_predict)
        return iterator

class PassthroughIterator(neukrill_net.image_directory_dataset.FlyIterator):
    def next(self):
        # check if we reached the end yet
        if self.final_iteration:
            raise StopIteration

        # allocate array
        if len(self.final_shape) == 2: 
            Xbatch = np.array(self.result.get(timeout=10.0)).reshape(
                        self.batch_size, self.final_shape[0],
                                         self.final_shape[1], 1)
        elif len(self.final_shape) == 3:
            Xbatch = np.array(self.result.get(timeout=10.0))
        # make sure it's float32
        Xbatch = Xbatch.astype(np.float32)

        if self.train_or_predict == "train":
            # get the batch for y as well
            ybatch = self.dataset.y[self.batch_indices,:].astype(np.float32)
        # index array for vbatch
        vbatch = self.dataset.cached[self.batch_indices,:]
        
        # start processing next batch
        if len(self.indices) >= self.batch_size:
            self.batch_indices = [self.indices.pop(0) 
                                for i in range(self.batch_size)]
            self.result = self.dataset.pool.map_async(self.dataset.fn,
                        [self.dataset.X[i] for i in self.batch_indices])
        else:
            self.final_iteration += 1

        if self.train_or_predict == "train":
            # get the batch for y as well
            return Xbatch,vbatch,ybatch
        elif self.train_or_predict == "test":
            return Xbatch,vbatch
        else:
            raise ValueError("Invalid option for train_or_predict:"
                    " {0}".format(self.train_or_predict))


class PassthroughDataset(neukrill_net.image_directory_dataset.ListDataset):
    """
    Dataset that can supply arbitrary vectors as well as the Conv2D
    spaces required by the convolutional layers.
    """
    def __init__(self, transformer, settings_path="settings.json", 
                 run_settings_path="run_settings/alexnet_based.json",
                 training_set_mode="train",
                 verbose=False, force=False, prepreprocessing=None,
                 cached=None):
        
        # runs inherited initialisation, but pulls out the
        # supplied cached array for iteration
        self.cached = sklearn.externals.joblib.load(cached).squeeze()
        
        # load settings
        # We don't save to self because super will do that
        settings = neukrill_net.utils.Settings(settings_path)
        run_settings = neukrill_net.utils.load_run_settings(run_settings_path,
                                                             settings,
                                                             force=force)
        
        # get the right split from run settings
        li = neukrill_net.utils.train_test_split_bool(settings.image_fnames,
                                    training_set_mode,
                                    train_split=run_settings['train_split'],
                                    classes=settings.classes)
        
        print '-----------'
        print training_set_mode
        print len(li)
        print sum(li)
        print '-----------'
        
        # Use boolean indexing
        self.cached = self.cached[li,:] 
        
        # Make sure our array has the correct 
        self.cached = self.cached.astype(np.float32)
        
        # may have to remove cached before handing it in...
        # ...no errors yet
        super(self.__class__,self).__init__(transformer=transformer, 
                 settings_path=settings_path, 
                 run_settings_path=run_settings_path,
                 training_set_mode=training_set_mode,
                 verbose=verbose, force=force,
                 prepreprocessing=prepreprocessing)
        
        
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
        iterator = PassthroughIterator(dataset=self, batch_size=batch_size, 
                                num_batches=num_batches, 
                                final_shape=self.run_settings["final_shape"],
                                rng=self.rng, mode=mode)
        return iterator

