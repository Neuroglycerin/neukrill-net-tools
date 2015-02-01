#!/usr/bin/env python
"""
Utility functions for the neukrill-net classifier
"""
import glob
import io
import json
import os
import gzip
import numpy as np

import neukrill_net.image_processing as image_processing
import neukrill_net.constants as constants

class Settings:
    """
    A class to handle all the parsing of the settings.json as well as
    checking the values and providing defaults
    """
    def __init__(self, settings_file):
        """
        Initialise settings object
        """
        self.required_settings = ['data_dir']

        # read the provided settings json (or stringIO)
        parsed_settings = self.parse_settings(settings_file)

        # check all required options have been set
        self.user_input = self.check_mandatory_fields(parsed_settings)

        # check the data dir options and find the correct one
        self.data_dir = self.check_data_dir(self.user_input['data_dir'])

        # check user input for random seed and if absent set it
        self.random_seed = self.user_input.get('r_seed', 42)

        # check user defined classes if absent set it
        self.classes = self.user_input.get('classes',
                                            constants.classes)


        # a way to encode the superclasses if we need to but don't want to complete
        # just now as it is very monotonous to copy everything
        # self.super_classes = {'FISH': ('fish classes'),
        #                       'DIATOMS': ('diatom classes'),
        #                       'TRICHODESMIUM': ('you get the idea')
        #                       'PROTISTS': ('protists_classes')
        #                       'NO_SUPER_CLASS': ('unclassified')}

        self._image_fnames = {}
        
        self._class_priors = []

    def parse_settings(self, settings_file):
        """
        Parse json file or stringIO formatted settings file
        This function will likely be extended later to enable defaults and to
        check inputs
        input : settings_file - filename or stringIO obj
                required_settings - list of required fields
        output: settings - settings dict
        """
        if settings_file.__class__ is io.StringIO:
            settings = json.load(settings_file)

        else:
            if not os.path.exists(settings_file):
                raise ValueError('Settings file does not exist: {0}'.format(\
                                                                settings_file))
            with open(settings_file, 'r') as settings_fh:
                settings = json.load(settings_fh)

        return settings


    def check_mandatory_fields(self, parsed_user_input):
        """
        Ensure all the mandatory settings are present
        """
        for entry in self.required_settings:
            if entry not in parsed_user_input:
                raise ValueError('data_dir must be defined')

        return parsed_user_input


    def check_data_dir(self, data_dir_possibilities):
        """
        Make sure the data dirs exist and resolve the test and train dirs
        abspaths
        """
        for possible_dir in data_dir_possibilities:
            train_data_dir = os.path.join(possible_dir, 'train')
            test_data_dir = os.path.join(possible_dir, 'test')
            dirs_exist = os.path.exists(train_data_dir) and \
                                                os.path.exists(test_data_dir)

            if dirs_exist:
                return os.path.abspath(possible_dir)

        raise ValueError("Can't find data dir in options: {0}".format(\
                                                            data_dir_possibilities))

    @property
    def image_fnames(self):
        """
        Take in data dir and return dict of filenames
        input: data_directory - path as str
        output: image_fnames - dict {'test': (tuple of fnames abspaths),
                                     'train': {'class_1' : (tuple of fnames),
                                               'class_2  : (tuple of fnames),
                                               ...
                                               }
                                    }
        """

        if not self._image_fnames:
            test_fnames = tuple(sorted(glob.glob(os.path.join(self.data_dir,
                                                       'test',
                                                       '*.jpg'))))

            # check there are the correct number of images
            num_test_images = len(test_fnames)
            if num_test_images != 130400:
                raise ValueError('Wrong number of test images found: {0}'
                                 ' instead of 130400'.format(num_test_images))

            train_fnames = {}

            for name in sorted(glob.glob(os.path.join(self.data_dir,
                                               'train',
                                               '*',
                                               ''))):
                split_name = name.split(os.path.sep)
                class_name = split_name[-2]
                image_names = sorted(glob.glob(os.path.join(name, '*.jpg')))
                train_fnames.update({class_name: image_names})

            num_train_classes = len(train_fnames.keys())
            num_train_images = sum(map(len, train_fnames.values()))
            if num_train_classes != 121:
                raise ValueError('Incorrect num of training class directories '\
                        '121 expected: {0} found'.format(num_train_classes))

            if num_train_images != 30336:
                raise ValueError('Incorrect num of training images '\
                        ' 30336 expected: {0} found'.format(num_train_images))

            self._image_fnames = {'test': test_fnames,
                                  'train': train_fnames}

        return self._image_fnames
    
    
    @property
    def class_priors(self):
        """
        Get the proportion of training data in each class
        """
        if self._class_priors == []:
            class_probs = np.zeros(len(self.classes))
            for class_index, class_name in enumerate(self.classes):
                # Tally up how many filenames are in this class
                # Set to zero if this class is not a field
                class_probs[class_index] = len(self.image_fnames['train'].get(class_name, 0))
            # Normalise the classes
            class_probs /= sum(class_probs)
            self._class_priors = class_probs
        
        return self._class_priors
        
        
def load_data(image_fname_dict, classes=None,
              processing=None, verbose=False):
    """
    Loads training or test data using image_processing.load_images func
    which applies the supplied processing function as required

    If the classes kwarg is not none assumed to be loading labelled train
    data and returns two np objs:
        * data - image matrix
        * labels - vector of labels

    if classes kwarg is none, data will be loaded as test data and just return
        * data - list of image vectors
    """

    if not processing and verbose:
        print("Warning: no processing applied, it will \
        not be possible to stack these images due to \
        varying sizes.")

    # initialise lists
    data = []

    # e.g. labelled training data
    if classes:
        labels = []

        for class_index, class_name in enumerate(classes):
            if verbose:
                print("class: {0} of 120: {1}".format(class_index, class_name))

            image_fpaths = image_fname_dict['train'][class_name]
            data_subset = image_processing.load_images(image_fpaths,
                                                          processing,
                                                          verbose)
            data.append(data_subset)
            num_images = len(data_subset)
            # generate the class labels and add them to the list
            array_labels = num_images * [class_name]
            labels = labels + array_labels
        return np.vstack(data), np.array(labels)

    # e.g. test data
    else:
        data = image_processing.load_images(image_fname_dict['test'],
                                            processing,
                                            verbose)
        names = [os.path.basename(fpath) for fpath in image_fname_dict['test']]
        return np.vstack(data), names


