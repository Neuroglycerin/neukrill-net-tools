"""
This is a DataSet class that will, given a root directory containing images
organised into folders with their class names, provide an interface to 
pylearn2 to load the images as required at runtime. At the same time, 
it will allow custom preprocessing functions to be run for augmenting the
data or otherwise.

Developed as part of our entry to the National Data Science Bowl plankton
classification challenge.
"""
__authors__ = "Gavin Gray"
__copyright__ = "Copyright 2015 - University of Edinburgh"
__credits__ = ["Gavin Gray"]
__license__ = "3-clause BSD"
__maintainer__ = "Gavin Gray"
__email__ = "gavingray1729@gmail.com"

############################################
#### Legacy imports - review ###############
############################################

import functools
#import logging
#import warnings
import numpy as np
from theano.compat.six.moves import xrange
from pylearn2.datasets import cache
from pylearn2.utils.iteration import (
    FiniteDatasetIterator,
    resolve_iterator_class
)

import copy
# Don't import tables initially, since it might not be available
# everywhere.
tables = None


from pylearn2.datasets.dataset import Dataset
from pylearn2.datasets import control
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace, IndexSpace
from pylearn2.utils import safe_zip
from pylearn2.utils.exc import reraise_as
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils import contains_nan
from theano import config


logger = logging.getLogger(__name__)


def ensure_tables():
    """
    Makes sure tables module has been imported
    """

    global tables
    if tables is None:
        import tables

############################################
#### Legacy imports - review ###############
############################################

import glob
import os

class ImageDirectory(Dataset):

    """
    This is class to provide an interface to a simple nested directory structure
    of images, where each directory in the root is a class and each image an 
    example, with options for how the preprocessing should be performed at 
    runtime to provide arrays in minibatches as expected.

    Parameters
    ----------
    X_paths : a tuple of the absolute paths of the images.
    y : optional - classes for each of the paths provided, as integers.
    preprocessing_settings : settings for the preprocessing functions. 
        This should be a dictionary of keyword strings as keys and 
        different options for the settings as a list.
   """
    _default_seed = (17, 2, 946)

    def __init__(self, X_paths, preprocessing_settings, rng=_default_seed, y=None):

        self.preprocessing_settings = preprocessing_settings
        self.X_paths = X_paths
        self.rng = rng

        # count the classes, if we can
        if y:
            self.y_labels = len(set(y_classes))
            # store y in one-hot encoding
            self.y = np.zeros((len(y),self.y_labels))
            for i,j in enumerate(y):
                self.y[i,j] = 1

        # load raw images as arrays
        images = []
        for image_path in X_paths:
            images.append(skimage.io.imread(image_path))

        # make tuple of preprocessing strings
        self.processing_strings = tuple(preprocessing_settings.keys())
        # prepare preprocessing options
        # could be problems with repeatability using this
        random.seed(rng)
        # lists of different settings
        self.settings = [preprocessing_settings[s] for s 
                in self.processing_strings]
        self.datapoints = self._random_product(self.X_paths,*self.settings)

    def _random_product(self,*args):
        """
        Simple implementation of random cartesian product sampling
        without replacement. 
        """
        # make all our pools into tuples
        pools = list(map(tuple,args))
        # iterate over all possible datapoints
        N_datapoints = np.prod(map(len,pools))
        # store all datapoints
        self._datapoint_history = []
        n = 0
        while n < N_datapoints:
            n += 1
            # propose a datapoint
            proposal = tuple(random.choice(pool) for pool in pools)
            # check we haven't picked it before
            if proposal not in self._datapoint_history:
                # store this datapoint
                self._datapoint_history.append(proposal)
                # and give it out
                yield proposal

    def iterator(self, batch_size=None, num_batches=None,
                 rng=None, data_specs=None, return_tuple=False):
        """
        Return an iterator for this dataset with the specified
        behaviour. Unspecified values are filled-in by the default.

        Parameters
        ----------
        mode : str or object, optional
            One of 'sequential', 'random_slice', or 'random_uniform',
            *or* a class that instantiates an iterator that returns
            slices or index sequences on every call to next().
            At the moment, will only have one mode.
        batch_size : int, optional
            The size of an individual batch. Optional if `mode` is
            'sequential' and `num_batches` is specified (batch size
            will be calculated based on full dataset size).
        num_batches : int, optional
            The total number of batches. Unnecessary if `mode` is
            'sequential' and `batch_size` is specified (number of
            batches will be calculated based on full dataset size).
        rng : int, object or array_like, optional
            Either an instance of `numpy.random.RandomState` (or
            something with a compatible interface), or a seed value
            to be passed to the constructor to create a `RandomState`.
            See the docstring for `numpy.random.RandomState` for
            details on the accepted seed formats. If unspecified,
            defaults to using the dataset's own internal random
            number generator, which persists across iterations
            through the dataset and may potentially be shared by
            multiple iterator objects simultaneously (see "Notes"
            below).
        return_tuple : bool, optional
            In case `data_specs` consists of a single space and source,
            if `return_tuple` is True, the returned iterator will return
            a tuple of length 1 containing the minibatch of the data
            at each iteration. If False, it will return the minibatch
            itself. This flag has no effect if data_specs is composite.
            Default: False.

        Returns
        -------
        iter_obj : object
            An iterator object implementing the standard Python
            iterator protocol (i.e. it has an `__iter__` method that
            return the object itself, and a `next()` method that
            returns results until it raises `StopIteration`).
            The `next()` method returns a batch containing data for
            each of the sources required in `data_specs`, in the requested
            `Space`.

        Notes
        -----
        Arguments are passed as instantiation parameters to classes
        that derive from `pylearn2.utils.iteration.SubsetIterator`.

        Iterating simultaneously with multiple iterator objects
        sharing the same random number generator could lead to
        difficult-to-reproduce behaviour during training. It is
        therefore *strongly recommended* that each iterator be given
        its own random number generator with the `rng` parameter
        in such situations.

        When it is valid to call the `iterator` method with the default
        value for all arguments, it makes it possible to use the `Dataset`
        itself as an Python iterator, with the default implementation of
        `Dataset.__iter__`. For instance, `DenseDesignMatrix` supports a
        value of `None` for `data_specs`.
        """
        # instantiate iterator and return it
        # THIS IS ON HOLD WHILE WORKING ON DENSE ALTERNATIVE


    def get_data(self):
        """
        Returns all the data, as it is internally stored.
        The definition and format of these data are described in
        `self.get_data_specs()`.

        Returns
        -------
        X : list of numpy arrays of different sizes.
            The data.
        or, if available:
        X,y : (list of numpy arrays of different sizes,
            numpy array of class labels, in one-hot encoding)
        """
        # this is probably a big problem for us

        if self.y is None:
            return self.X
        else:
            return (self.X, self.y)

    def get_topo_batch_axis(self):
        """
        The index of the axis of the batches

        Returns
        -------
        axis : int
            The axis of a topological view of this dataset that corresponds
            to indexing over different examples.
        """
        # Going to assume we end up with 3d array, and the 3rd dimension will
        # be different examples.
        return 2

    def get_targets(self):
        """
        Returns the labels in one hot encoding.
        """
        return self.y

    def get_batch_design(self, batch_size, include_labels=False):
        """
        Returns a randomly chosen batch of data formatted as a design
        matrix.

        This method is not guaranteed to have any particular properties
        like not repeating examples, etc. It is mostly useful for getting
        a single batch of data for a unit test or a quick-and-dirty
        visualization. Using this method for serious learning code is
        strongly discouraged. All code that depends on any particular
        example sampling properties should use Dataset.iterator.

        .. todo::

            Refactor to use `include_targets` rather than `include_labels`,
            to make the terminology more consistent with the rest of the
            library.

        Parameters
        ----------
        batch_size : int
            The number of examples to include in the batch.
        include_labels : bool
            If True, returns the targets for the batch, as well as the
            features.

        Returns
        -------
        batch : member of feature space, or member of (feature, target) space.
            Either numpy value of the features, or a (features, targets) tuple
            of numpy values, depending on the value of `include_labels`.
        """

    def get_batch_topo(self, batch_size, include_labels=False):
        """
        Returns a topology-preserving batch of data.

        This method is not guaranteed to have any particular properties
        like not repeating examples, etc. It is mostly useful for getting
        a single batch of data for a unit test or a quick-and-dirty
        visualization. Using this method for serious learning code is
        strongly discouraged. All code that depends on any particular
        example sampling properties should use Dataset.iterator.

        .. todo::

            Refactor to use `include_targets` rather than `include_labels`,
            to make the terminology more consistent with the rest of the
            library.

        Parameters
        ----------
        batch_size : int
            The number of examples to include in the batch.
        include_labels : bool
            If True, returns the targets for the batch, as well as the
            features.

        Returns
        -------
        batch : member of feature space, or member of (feature, target) space.
            Either numpy value of the features, or a (features, targets) tuple
            of numpy values, depending on the value of `include_labels`.
        """

    @functools.wraps(Dataset.get_num_examples)
    def get_num_examples(self):
        """
        Returns the number of examples, should be possible to estimate from
        number of examples and possible combinations of preprocessing functions.

        Otherwise, just return float('inf').
        """
        return float('inf')

    def has_targets(self):
        """
        Returns true if we've got targets.
        """
        return self.y is not None

    ### restrict method could go here, as in DenseDesignMatrix ###

     #def get_data_specs(self):
     #   """
     #   Returns the data_specs specifying how the data is internally stored.
#
#        This is the format the data returned by `self.get_data()` will be.
#        """
#        return self.data_specs

