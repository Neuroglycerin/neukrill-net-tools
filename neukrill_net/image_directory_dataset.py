"""
This is a DataSet class that will, given a root directory containing images
organised into folders with their class names, provide an interface to 
pylearn2 to load the images at start time. Then, at runtime it will allow 
custom preprocessing functions to be run for augmenting the data or 
otherwise.

Developed as part of our entry to the National Data Science Bowl plankton
classification challenge.
"""
__authors__ = "Gavin Gray"
__copyright__ = "Copyright 2015 - University of Edinburgh"
__credits__ = ["Gavin Gray"]
__license__ = "3-clause BSD"
__maintainer__ = "Gavin Gray"
__email__ = "gavingray1729@gmail.com"


import numpy as np

import pylearn2.datasets.dataset 
import neukrill_net.utils
import encoding as enc

# don't have to think too hard about how to write this:
# https://stackoverflow.com/questions/19151/build-a-basic-python-iterator
class FlyIterator(object):
    """
    Simple iterator class to take a dataset and iterate over
    it's contents applying a processing function. Assuming
    the dataset has a processing function to apply.
    
    It may have an issue of there being some leftover examples
    that will never be shown on any epoch. Can avoid this by
    seeding with sampled numbers from the dataset's own rng.
    """
    def __init__(self, dataset, batch_size, num_batches,
                 final_shape, rng, mode='even_shuffled_sequential'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.final_shape = final_shape
        # initialise rng
        self.rng = rng
        # shuffle indices of size equal to number of examples
        # in dataset
        # indices of size equal to number of examples
        N = self.dataset.get_num_examples()
        self.indices = range(N)
        if mode == 'even_shuffled_sequential':
            self.rng.shuffle(self.indices)
        else:
            assert mode == 'even_sequential'
        # have to add this for checks during training
        # bit of a lie
        self.stochastic = False
        # this is also required
        self.num_examples = batch_size*num_batches
        
    def __iter__(self):
        return self
    
    def next(self):
        # return one batch
        if len(self.indices) >= self.batch_size:
            batch_indices = [self.indices.pop(0) for i in range(self.batch_size)]
            # preallocate array
            if len(self.final_shape) == 2: 
                Xbatch = np.zeros([self.batch_size]+list(self.final_shape)+[1])
            elif len(self.final_shape) == 3:
                Xbatch = np.zeros([self.batch_size]+list(self.final_shape))
            # iterate over indices, applying the dataset's processing function
            for i,j in enumerate(batch_indices):
                Xbatch[i] = self.dataset.fn(self.dataset.X[j]).reshape(Xbatch.shape[1:])
            # get the batch for y as well
            ybatch = self.dataset.y[batch_indices,:].astype(np.float32)
            Xbatch = Xbatch.astype(np.float32)
            return Xbatch,ybatch
        else:
            raise StopIteration


class ListDataset(pylearn2.datasets.dataset.Dataset):
    """
    Loads images as raw numpy arrays in a list, tries 
    its best to respect the interface expected of a 
    Pylearn2 Dataset.
    """
    def __init__(self, transformer, settings_path="settings.json", 
                 run_settings_path="run_settings/alexnet_based.json",
                 training_set_mode="train",
                 verbose=False, force=False):
        """
        Loads the images as a list of differently shaped
        numpy arrays and loads the labels as a vector of 
        integers, mapped deterministically.
        """
        self.fn = transformer
        # load settings
        self.settings = neukrill_net.utils.Settings(settings_path)
        self.run_settings = neukrill_net.utils.load_run_settings(run_settings_path,
                                                                 self.settings,
                                                                 force=force)
        # split train/test/validation
        self.settings.image_fnames["train"] = \
                neukrill_net.utils.train_test_split(
                        self.settings.image_fnames, 
                        training_set_mode, 
                        train_split=self.run_settings["train_split"])
        #load the data
        self.X, labels = neukrill_net.utils.load_rawdata(
                        self.settings.image_fnames,
                        classes=self.settings.classes,
                        verbose=verbose)
        self.N = len(self.X)
        # count the classes
        self.n_classes = len(self.settings.classes)
        # transform labels from strings to integers
        if self.run_settings.get("use_super_classes", False):
            supclass_vecs = {}
            general_hier = enc.get_hierarchy()
            lengths = sum([len(array) for array in hier])
            self.y = np.zeros((self.N,lengths)
        else:
            self.y = np.zeros((self.N,self.n_classes))
        class_dictionary = {}
        for i,c in enumerate(self.settings.classes):
            class_dictionary[c] = i
        for i,j in enumerate(map(lambda c: class_dictionary[c],labels)):
            if self.run_settings.get("use_super_classes", False):
                if not supclass_vecs.has_key(class_label):
                    supclass_hier = enc.get_encoding(class_label, general_hier)
                    supclass_vecs[class_label] = \
                                [el for grp in supclass_hier for el in grp]
                    y[i,:] = np.array(supclass_vecs[class_label])
            else:
                self.y[i,j] = 1
        self.y = self.y.astype(np.float32)
        
        # set up the random state
        self.rng = np.random.RandomState(self.settings.random_seed)
        
        # shuffle a list of image indices
        self.indices = range(self.N)
        self.rng.shuffle(self.indices)
        
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
        iterator = FlyIterator(dataset=self, batch_size=batch_size, 
                                num_batches=num_batches, 
                                final_shape=self.run_settings["final_shape"],
                                rng=self.rng)
        return iterator
        
    def adjust_to_be_viewed_with():
        raise NotImplementedError("Didn't think this was important, so didn't write it.")
    
    def get_batch_design(self, batch_size, include_labels=False):
        """
        Will return a list of the size batch_size of carefully raveled arrays.
        Optionally, will also include labels (using include_labels).
        """
        selection = self.rng.random_integers(0,high=self.N,size=batch_size)
        batch = [self.X[s].ravel() for s in selection]
        if include_labels:
            raise NotImplementedError
        return batch
        
    def get_batch_topo(self, batch_size, include_labels=False):
        """
        Will return a list of the size batch_size of raw, unfiltered, artisan
        numpy arrays. Optionally, will also include labels (using include_labels).
        
        Strongly discouraged to use this method for learning code, so I guess 
        this isn't so important?
        """
        selection = self.rng.random_integers(0,high=self.N,size=batch_size)
        batch = [self.X[s] for s in selection]
        if include_labels:
            raise NotImplementedError
        return batch
        
    def get_num_examples(self):
        return self.N
        
    def get_topological_view():
        raise NotImplementedError("Not written yet, not sure we need it")
        
    def get_weights_view():
        raise NotImplementedError("Not written yet, didn't think it was important")
        
    def has_targets(self):
        if self.y:
            return True
        else:
            return False
