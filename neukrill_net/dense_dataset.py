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
            train_or_predict="train", verbose=False, force=False):
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
                    self.train_test_split(train_or_predict, training_set_mode)

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
            for class_label in self.settings.classes:
                for image_path in self.settings.image_fnames[
                                                    train_or_predict][class_label]:
                    # load the image as numpy array
                    image = skimage.io.imread(image_path)
                    # apply processing function (get back multiple images)
                    images = processing(image)
                    # for each image store a class label
                    y += [class_label]*len(images)
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
            # make sure y is an array
            y = np.array(y)
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
            X = np.zeros((self.N_images,self.run_settings["final_shape"][0],
                self.run_settings["final_shape"][1],1))
            image_index = 0
            # loop over all the images, in order
            for image_path in self.settings.image_fnames[train_or_predict]:
                # load the image as numpy array
                image = skimage.io.imread(image_path)
                # apply processing function (get back multiple images)
                images = processing(image)
                # then broadcast each of these images into the empty X array
                for image in images:
                    X[image_index,:,:,0] = image
                    image_index += 1
            # store the names in this dataset object
            self.names = [os.path.basename(fpath) for fpath in 
                    self.settings.image_fnames[train_or_predict]]

            # now run inherited initialisation
            super(self.__class__,self).__init__(topo_view=X)
        else:
            raise ValueError('Invalid option: should be either "train" for'
                             'training or "test" for prediction (I know '
                             ' that is annoying).')


    def train_test_split(self, train_or_predict, training_set_mode):
        """
        Perform a stratified split of the image paths stored in
        settings. Iterates over all class labels to do this.
        """
        # stratified split of the image paths for train, validation and test
        # iterate over classes, removing some proportion of the elements, in a 
        # deterministic way
        train_split = self.run_settings["train_split"]
        test_split = train_split + (1-train_split)/2
        # initialise new variable to store split
        image_fnames = {}
        # assuming train split is some float between 0 and 1, and assign that
        # proportion to train and half of the remaining to test and validation
        for class_label in self.settings.classes:
            # find where the break should be
            train_break = int(train_split*len(
                self.settings.image_fnames[train_or_predict][class_label]))
            test_break = int(test_split*len(
                self.settings.image_fnames[train_or_predict][class_label]))
            if training_set_mode == "train":
                # then everything up to train_break is what we want
                image_fnames[class_label] \
                        = self.settings.image_fnames\
                        [train_or_predict][class_label][:train_break]
            elif training_set_mode == "validation":
                # then we want the _first_ half of everything after train_break
                image_fnames[class_label] \
                        = self.settings.image_fnames \
                        [train_or_predict][class_label][train_break:test_break]
            elif training_set_mode == "test":
                # then we want the _second_ half of everything after train_break
                image_fnames[class_label] \
                        = self.settings.image_fnames \
                        [train_or_predict][class_label][test_break:]
            else:
                raise ValueError("Invalid option for training set mode.")
        return image_fnames
