#!/usr/bin/env python
"""
Utility functions for the neukrill-net classifier
"""
import glob
import hashlib
import io
import json
import os
import sys

def parse_settings(settings_file):
    """
    Parse json file or stringIO formatted settings file
    This function will likely be extended later to enable defaults and to
    check inputs
    input : settings_file - filename or stringIO obj
    output: settings - settings dict
    """
    if settings_file.__class__ is io.StringIO:
        settings = json.load(settings_file)

    else:
        with open(settings_file, 'r') as settings_fh:
            settings = json.load(settings_fh)

    required_settings = ['data_dir']

    for entry in required_settings:
        if entry not in settings:
            raise ValueError('data_dir must be defined')

    settings = check_data_dir(settings)

    return settings


def check_data_dir(settings):
    """
    Make sure the data dirs exist and resolve the test and train dirs abspaths
    """
    # remove in case it has been accidentally added to settings
    if 'test_data_dir' in settings:
        del settings['test_data_dir']

    # go through list of possible data dirs in settings and check them
    for possible_dir in settings['data_dir']:
        train_data_dir = os.path.join(possible_dir, 'train')
        test_data_dir = os.path.join(possible_dir, 'test')
        dirs_exist = os.path.exists(train_data_dir) and os.path.exists(test_data_dir)

        if dirs_exist:
            settings.update({'train_data_dir': os.path.abspath(train_data_dir),
                             'test_data_dir': os.path.abspath(test_data_dir),
                             'data_dir': os.path.abspath(possible_dir)})
            break


    if 'train_data_dir' not in settings:
        raise ValueError('No data dir not found')

    return settings

def load_images(image_fname_dict, processing=None, verbose=False):
    """Loads images and applies a processing
    function if supplied one.
    
    Processing function is expected to take a 
    single argument, the image as a numpy array,
    and process it.
    
    Returns two lists:
        * data - list of image vectors
        * labels - list of labels"""
    if not processing and verbose:
        print("Warning: no processing applied, it will 
        not be possible to stack these images due to
        varying sizes.")

    # initialise lists
    data = []
    labels = []
    class_label_list = []
    for class_index, class_name in enumerate(image_fname_dict.keys()):
        if verbose:
            print("class: {0} of 120: {1}".format(class_index, class_name))
        image_fpaths = image_fname_dict[class_name]
        num_image = len(image_fpaths)
        #image_array = np.zeros((num_image, 625))

        class_label_list.append(class_name)
        for index in range(num_image):
            # read the image into a numpy array
            image = skimage.io.imread(image_fpaths[index])
            
            if processing:
                resized_image = processing(image)
                image_vector = resized_image.ravel()
            else:
                image_vector = image.ravel()

            #image_array[index,] = image_vector
            data.append(image_vector)

        # generate the class labels and add them to the list
        array_labels = num_image * [class_name]
        labels = labels + array_labels

    return data, labels
