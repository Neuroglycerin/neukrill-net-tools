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
__license__ = "MIT"
__maintainer__ = "Gavin Gray"
__email__ = "gavingray1729@gmail.com"


import numpy as np

import pylearn2.datasets.dataset 
import neukrill_net.utils
import neukrill_net.encoding
import neukrill_net.image_processing
import skimage.util
import multiprocessing

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
                 final_shape, rng, mode='even_shuffled_sequential',
                 train_or_predict="train", n_jobs=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.final_shape = final_shape
        self.train_or_predict = train_or_predict
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
        
        # prepare first batch
        self.batch_indices = [self.indices.pop(0) for i in range(self.batch_size)]
        # start the first batch computing
        self.result = self.dataset.pool.map_async(self.dataset.fn,
                            [self.dataset.X[i] for i in self.batch_indices])

        # initialise flag variable for final iteration
        self.final_iteration = 0
        
    def __iter__(self):
        return self
    
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
            return Xbatch,ybatch
        elif self.train_or_predict == "test":
            return Xbatch
        else:
            raise ValueError("Invalid option for train_or_predict:"
                    " {0}".format(self.train_or_predict))


class ListDataset(pylearn2.datasets.dataset.Dataset):
    """
    Loads images as raw numpy arrays in a list, tries 
    its best to respect the interface expected of a 
    Pylearn2 Dataset.
    """
    def __init__(self, transformer, settings_path="settings.json", 
                 run_settings_path="run_settings/alexnet_based.json",
                 training_set_mode="train", train_or_predict="train",
                 verbose=False, force=False, prepreprocessing=None,
                 n_jobs=1):
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
        self.train_or_predict = train_or_predict
        # initialise pool
        self.pool = multiprocessing.Pool(n_jobs)
        #import pdb
        #pdb.set_trace()

        if train_or_predict == "train":
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
            self.n_classes = len(self.settings.classes)

            if self.run_settings.get("use_super_classes", False):
                supclass_vecs = {}
                general_hier = neukrill_net.encoding.get_hierarchy(self.settings)
                n_columns = sum([len(array) for array in general_hier])
                self.y = np.zeros((self.N,n_columns))
                class_dictionary = neukrill_net.encoding.make_class_dictionary(
                                            self.settings.classes,general_hier)
            else:
                self.y = np.zeros((self.N,self.n_classes))
                class_dictionary = {}
                for i,c in enumerate(self.settings.classes):
                    class_dictionary[c] = i
            for i,j in enumerate(map(lambda c: class_dictionary[c],labels)):
                self.y[i,j] = 1
            self.y = self.y.astype(np.float32)

        elif train_or_predict == "test":
            self.X, self.names = neukrill_net.utils.load_rawdata(
                            self.settings.image_fnames,
                            verbose=verbose)

        self.N = len(self.X)
        
        if prepreprocessing is not None:
            self.X = [neukrill_net.image_processing.resize_image(skimage.util.img_as_float(image),
                                    prepreprocessing['resize'],
                                    order=prepreprocessing['resize_order'])
                                    for image in self.X]
        
        # count the classes
        self.n_classes = len(self.settings.classes)
        
        # set up the random state
        self.rng = np.random.RandomState(self.settings.random_seed)
        
    def iterator(self, mode=None, batch_size=None, num_batches=None, rng=None,
                        data_specs=None, return_tuple=False):
        """
        Returns iterator object with standard Pythonic interface; iterates
        over the dataset over batches, popping off batches from a shuffled 
        list of indices.
        Inputs:
            - mode: 'even_sequential' or 'even_shuffled_sequential'.
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
                                rng=self.rng, mode=mode,
                                train_or_predict=self.train_or_predict)
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
