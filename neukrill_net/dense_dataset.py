"""
This is intended to be a simple inheritance of the DenseDesignMatrix class
of pylearn2. The initialisation has been modified to use a settings file
to load the images and process them before running the original initialisation.
"""
__authors__ = "Gavin Gray"
__copyright__ = "Copyright 2015 - University of Edinburgh"
__credits__ = ["Gavin Gray"]
__license__ = "3-clause BSD"
__maintainer__ = "Gavin Gray"
__email__ = "gavingray1729@gmail.com"

import os
import json
import skimage.io
import neukrill_net.utils
import neukrill_net.augment
import pylearn2.datasets
import numpy as np
import sys
import encoding as enc

from pylearn2.datasets.dataset import Dataset
import functools
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace, IndexSpace
from pylearn2.utils import safe_zip
from pylearn2.utils.iteration import (
    FiniteDatasetIterator,
    resolve_iterator_class
)

class DensePNGDataset(pylearn2.datasets.DenseDesignMatrix):
    """
    A class intended to load images from a directory, apply some
    processing and initialise a pylearn2 DenseDesignmatrix model.

    Parameters
    ----------
    settings_path : settings.json file path used to find images and control
        how they are loaded and processed.
    run_settings : another json file used to control settings for individual
        runs.
    training_set_mode : string option controlling whether this object will be
        for a training set, validation set or test set. Not to be confused 
        with when we're dealing with a prediction set, where we don't know 
        the labels (see next option). Would consider better name for this
        variable.
    train_or_predict : depending on whether we're training a model or 
        predicting on the supplied test set. For training, use: "train". 
        At the risk of confusion, for prediction use keystring: "test"
    """
    def __init__(self,settings_path="settings.json",
            run_settings="run_settings/default.json",training_set_mode="train",
            train_or_predict="train", verbose=False, force=False, split=1):
        # parse the settings file
        self.settings = neukrill_net.utils.Settings(settings_path)
        # get the run settings
        if train_or_predict == 'test':
            force=True
        self.run_settings = neukrill_net.utils.load_run_settings(run_settings,
                                                                self.settings,
                                                                force=force)
        processing_settings = self.run_settings["preprocessing"]
        # get a processing function from this
        processing = neukrill_net.augment.augmentation_wrapper(
                                                        **processing_settings)

        # super simple if statements for predict/train
        if train_or_predict == "train":
            # split the dataset based on training_set_mode option:
            self.settings.image_fnames[train_or_predict] = \
                    neukrill_net.utils.train_test_split(
                            self.settings.image_fnames, 
                            training_set_mode, 
                            train_split=self.run_settings["train_split"])

            # count the images
            self.N_images = sum(1 for class_label in self.settings.classes
                    for image_path in 
                    self.settings.image_fnames[train_or_predict][class_label])
            # multiply that count by augmentation factor
            self.N_images = int(self.N_images*
                    self.run_settings["augmentation_factor"])
            # initialise y vector
            y = []
            # initialise array
            X = np.zeros((self.N_images,self.run_settings["final_shape"][0],
                self.run_settings["final_shape"][1],1))
            image_index = 0
            # load the images in image_fpaths, iterating and keeping track of class
            if self.run_settings.get("use_super_classes", False):
                # create dictionary to cache superclass vectors
                supclass_vecs = {}
                # get the general hierarchy
                general_hier = enc.get_hierarchy(self.settings)
            for class_label in self.settings.classes:
                for image_path in self.settings.image_fnames[
                                                    train_or_predict][class_label]:
                    # load the image as numpy array
                    image = skimage.io.imread(image_path)
                    # apply processing function (get back multiple images)
                    images = processing(image)
                    # for each image store a class label
                    if self.run_settings.get("use_super_classes", False):
                        # check if superclass vector for this class label
                        # already generated, if not generate
                        if not supclass_vecs.has_key(class_label):
                            # get superclass hierarchy for class label
                            supclass_hier = enc.get_encoding(class_label, general_hier)
                            # collapse to a list of 1/0 values
                            supclass_vecs[class_label] = \
                                [el for grp in supclass_hier for el in grp]
                        y += [supclass_vecs[class_label]]*len(images)
                    else:
                        y += [class_label]*len(images)
                    # then broadcast each of these images into the empty X array
                    for image in images:
                        X[image_index,:,:,0] = image
                        image_index += 1
            # if we're normalising
            if processing_settings.get("normalise", False):
                if verbose:
                    print("Applying normalisation: {0}".format(
                        processing_settings["normalise"]["global_or_pixel"]))
                # then call the normalise function
                X,self.run_settings = neukrill_net.image_processing.normalise(X,
                                            self.run_settings, verbose=verbose)
            # make sure y is an array
            y = np.array(y)
            if self.run_settings.get("use_super_classes", False):
                # using superclasses so y already contains target vectors
                super(self.__class__,self).__init__(topo_view=X,y=y)
            else:
                # not using superclasses so map label strings to integers
                # count the y labels
                N_y_labels = len(list(set(y)))
                # build dictionary to encode labels numerically
                class_dictionary = {}
                for i,c in enumerate(self.settings.classes):
                    class_dictionary[c] = i
                # map to integers
                y = np.array(map(lambda c: class_dictionary[c], y))
                # make it 2D column vector
                y = y[np.newaxis].T
                # now run inherited initialisation
                super(self.__class__,self).__init__(topo_view=X,y=y,y_labels=N_y_labels)
                

        elif train_or_predict == "test":
            # split test paths if we're splitting them
            self.settings.image_fnames = neukrill_net.utils.test_split(split, 
                    self.settings.image_fnames)

            # test is just a big list of image paths
            # how many?
            self.N_images = sum(1 for image_path in 
                    self.settings.image_fnames[train_or_predict])
            # check augmentation in the traditional way (it's boilerplate time)
            self.N_images = int(self.N_images*
                    self.run_settings["augmentation_factor"])

            # more boilerplate code, but it's going to be easier than making a
            # function that can deal with the above as well
            # initialise array
            #import pdb
            #pdb.set_trace()
            X = np.zeros((self.N_images,self.run_settings["final_shape"][0],
                self.run_settings["final_shape"][1],1))
            image_index = 0
            if verbose:
                print("Loading this many images:...........................")
                # get a list of 50 image_paths to watch out for
                stepsize = int(len(self.settings.image_fnames[train_or_predict])/50)
                progress_paths = [impath for i,impath in 
                        enumerate(self.settings.image_fnames[train_or_predict]) 
                        if i%stepsize == 0 ]
            # loop over all the images, in order
            for image_path in self.settings.image_fnames[train_or_predict]:
                if verbose:
                    if image_path in progress_paths: 
                        sys.stdout.write(".")
                        sys.stdout.flush()
                        # if it's the last one we better stick a newline on
                        if image_path == progress_paths[-1]:
                            sys.stdout.write(".\n")
                # load the image as numpy array
                image = skimage.io.imread(image_path)
                # apply processing function (get back multiple images)
                images = processing(image)
                # then broadcast each of these images into the empty X array
                for image in images:
                    X[image_index,:,:,0] = image
                    image_index += 1
            # if we're normalising
            if processing_settings.get("normalise",0):
                if verbose:
                    print("Applying normalisation: {0}".format(
                        processing_settings["normalise"]["global_or_pixel"]))
                # then call the normalise function
                X,self.run_settings = neukrill_net.image_processing.normalise(X,
                                            self.run_settings, verbose=verbose)
            # store the names in this dataset object
            self.names = [os.path.basename(fpath) for fpath in 
                    self.settings.image_fnames[train_or_predict]]

            # now run inherited initialisation
            super(self.__class__,self).__init__(topo_view=X)
        else:
            raise ValueError('Invalid option: should be either "train" for'
                             'training or "test" for prediction (I know '
                             ' that is annoying).')

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 rng=None, data_specs=None,
                 return_tuple=False):
        """
        Copied from dense_design_matrix, in order to fix uneven problem.
        """

        if data_specs is None:
            data_specs = self._iter_data_specs

        # If there is a view_converter, we have to use it to convert
        # the stored data for "features" into one that the iterator
        # can return.
        space, source = data_specs
        if isinstance(space, CompositeSpace):
            sub_spaces = space.components
            sub_sources = source
        else:
            sub_spaces = (space,)
            sub_sources = (source,)

        convert = []
        for sp, src in safe_zip(sub_spaces, sub_sources):
            if src == 'features' and \
               getattr(self, 'view_converter', None) is not None:
                conv_fn = (lambda batch, self=self, space=sp:
                           self.view_converter.get_formatted_batch(batch,
                                                                   space))
            else:
                conv_fn = None

            convert.append(conv_fn)

        # TODO: Refactor
        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)

        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if rng is None and mode.stochastic:
            rng = self.rng
        # hack to make the online augmentations run
        FiniteDatasetIterator.uneven = False
        iterator = FiniteDatasetIterator(self,
                                 mode(self.X.shape[0],
                                      batch_size,
                                      num_batches,
                                      rng),
                                 data_specs=data_specs,
                                 return_tuple=return_tuple,
                                 convert=convert)
        return iterator
