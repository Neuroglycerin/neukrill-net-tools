#!/usr/bin/env python
"""
Utility functions for the neukrill-net classifier
"""
import glob
import io
import json
import os
import csv
import gzip
import numpy as np
import skimage
import random
import warnings
import sklearn.externals

import neukrill_net.image_processing as image_processing
import neukrill_net.constants as constants
import neukrill_net.taxonomy
import neukrill_net.encoding

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
        # don't shuffle by default
        self.shuffle_flag = False

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
            # expand out any environment variables in paths
            possible_dir = os.path.expandvars(possible_dir)
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
                if self.shuffle_flag:
                    random.seed(self.random_seed)
                    # shuffle in place
                    random.shuffle(image_names)
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

    def shuffle(self, seed=None):
        """
        Modifies processing of image_fnames to ensure that the paths inside
        each class will be shuffled. (Internally, wipes existing data
        structure and rewrites it with shuffling on next call.
        Input:
            seed: random seed to use for shuffling, default is 42.
        Output:
            None (sets internal flag)
        """
        self.shuffle_flag = True
        # if seed supplied, apply new value
        if seed:
            self.random_seed = seed
        # if image_fnames exists, get rid of it
        if self._image_fnames:
            self._image_fnames = None
        return None


    def flattened_train_paths(self, class_names):
        """
        Flattens the training paths
        Input : class_names
        Output: X - a list of the flattened training data paths
                y - a list of the class labels (index in provided list) for each path
        """
        num_paths = sum([len(self.image_fnames['train'][classname]) for classname in self.image_fnames['train'].keys()])
        paths = []
        labels = []
        for class_index, class_name in enumerate(class_names):
            paths += self.image_fnames['train'][class_name]
            labels += len(self.image_fnames['train'][class_name]) * [class_index]
        return paths, labels


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
    which applies the supplied processing function as required.

    This function is rapidly becoming a hack and will probably be replace with
    Scott's HighLevelFeatures idea.

    If the classes kwarg is not none assumed to be loading labelled train
    data and returns two np objs:
        * data - image matrix
        * labels - vector of labels

    if classes kwarg is none, data will be loaded as test data and just return
        * data - list of image vectors
    """

    if not processing:
        if verbose:
            print("Warning: no processing applied, it will \
            not be possible to stack these images due to \
            varying sizes.")
        # use the raw loading without processing
        return load_rawdata(image_fname_dict, classes=classes, verbose=verbose)

    # check augmentation factor
    dummy_images = processing(np.zeros((100,100)))
    if type(dummy_images) is not list:
        dummy_images = [dummy_images]
    if dummy_images[0].shape == ():
        dummy_images[0] = np.array(dummy_images[0])[np.newaxis]
    augmentation_factor = len(dummy_images)
    expected_shape = dummy_images[0].shape
    # second dummy images
    dummy_images = processing(np.zeros((42,42)))
    if type(dummy_images) is not list:
        dummy_images = [dummy_images]
    if dummy_images[0].shape == ():
        dummy_images[0] = np.array(dummy_images[0])[np.newaxis]
    # use this opportunity to check if we're going to be able to process these
    if expected_shape != dummy_images[0].shape:
        # then we'll just load them as a list and process them after
        X,labels = load_rawdata(image_fname_dict, classes=classes, verbose=verbose)
        # reprocess X
        X_new = []
        labels_new = []
        for image,label in zip(X,labels):
            p_images = processing(image)
            if type(p_images) is not list:
                p_images = [p_images]
            X_new += p_images
            labels_new += len(p_images)*[label]
        return X_new,labels_new

    # e.g. labelled training data
    if classes:
        labels = []

        image_fpaths = []

        for class_index, class_name in enumerate(classes):
            if verbose:
                print("class: {0} of 120: {1}".format(class_index, class_name))

            class_fpaths = image_fname_dict['train'][class_name]
            array_labels = augmentation_factor * len(class_fpaths) * [class_name]
            labels = labels + array_labels
            image_fpaths += class_fpaths
            #data_subset = image_processing.load_images(image_fpaths,
            #                                              processing,
            #                                              verbose)
            #data.append(data_subset)
            #num_images = len(data_subset)
            # generate the class labels and add them to the list
        data = image_processing.load_images(image_fpaths, processing, verbose)
        return data, np.array(labels)

    # e.g. test data
    else:
        data = image_processing.load_images(image_fname_dict['test'],
                                            processing,
                                            verbose)
        names = [os.path.basename(fpath) for fpath in image_fname_dict['test']]
        return np.vstack(data), names


def load_rawdata(image_fname_dict, classes=None, verbose=False):
    """
    Loads training or test data without appyling any processing.

    If the classes kwarg is not none assumed to be loading labelled train
    data and returns two np objs:
        * data - list of image matrices
        * labels - vector of labels

    if classes kwarg is none, data will be loaded as test data and just return
        * data - list of image matrices
    """

    # initialise lists
    data = []

    # e.g. labelled training data
    if classes:
        labels = []

        for class_index, class_name in enumerate(classes):
            if verbose:
                print("class: {0} of 120: {1}".format(class_index, class_name))

            fpaths = image_fname_dict['train'][class_name]

            # Load the data and add to list
            data += [skimage.io.imread(fpath) for fpath in fpaths]

            # generate the class labels and add them to the list
            labels += len(fpaths) * [class_name]

        return data, np.array(labels)

    # e.g. test data
    else:
        data = [skimage.io.imread(fpath) for fpath in image_fname_dict['test']]
        names = [os.path.basename(fpath) for fpath in image_fname_dict['test']]
        return data, names


def write_predictions(out_fname, p, names, classes):
    """
    Write probabilities to an output csv which is compressed with gzip.
    input:  out_fname - name of the output file.
                       Append .csv if you like. Do not append with .gz.
            p - probability matrix.
                dim-0 specifies the file predicitons are of
                dim-1 specifies which class the prediciton is for
            names - names of the things which are being classified
            classes - the classes predictions are placed into
    output: None
            writes a gzip compressed csv file to `out_fname`.gz on disk
    """

    # Write the probabilites as a CSV
    with open(out_fname, 'w') as csv_out:
        out_writer = csv.writer(csv_out, delimiter=',')
        out_writer.writerow(['image'] + list(classes))
        for index in range(len(names)):
            out_writer.writerow([names[index]] + list(p[index,]))

    # Compress with gzip
    with open(out_fname, 'rb') as f_in:
        f_out = gzip.open(out_fname + '.gz', 'wb')
        f_out.writelines(f_in)
        f_out.close()

    # Delete the uncompressed CSV
    os.unlink(out_fname)


def load_run_settings(run_settings_path, settings,
        settings_path="settings.json", verbose=False, force=False):
    """
    Loads the run settings and adds settings to dictionary, along
    with:
    * filename - the filename minus ".json" of the run settings file, for
        saving results, pickles, etc.
    * run_settings_path - abspath to the run settings file, is handed
        to the Dataset class for Pylearn2
    * settings_path - abspath to the settings file, assumed to be in the usual
        cwd
    * modeldir - directory in which to save models
    * pickle abspath - abspath where _this_ run will save its pickle file
    """

    with open(run_settings_path) as rf:
        run_settings = json.load(rf)
    # shoehorn run_settings filename into its own dictionary (for later)
    run_settings['filename'] = os.path.split(
                                        run_settings_path)[-1].split(".")[0]
    # and the full run settings path
    run_settings['run_settings_path'] = os.path.abspath(run_settings_path)
    # if shuffling is specified and it's 1
    if run_settings.get('shuffle',0):
        # shuffle the dataset and apply seed if we have one
        settings.shuffle(run_settings.get('random seed',None))
    # also put the settings in there
    run_settings['settings'] = settings
    # and the settings path, while we're at it
    run_settings['settings_path'] = os.path.abspath(settings_path)
    # add the models directory for this run
    # before saving model check there is somewhere for it to save to
    modeldir = os.path.join(settings.data_dir,"models")
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)
    run_settings['modeldir'] = modeldir

    # save the pickle name here, so it's less likely to get garbled between train and test
    run_settings['pickle abspath'] = os.path.join(modeldir,run_settings['filename']+".pkl")
    # also want alternative save path in some situations
    run_settings['alt_picklepath'] = os.path.join(modeldir,run_settings['filename']+"_recent.pkl")
    # check if the pickle already exists - and don't allow overwriting if so
    if os.path.exists(run_settings['pickle abspath']) and not force:
        # not sure what type of error this should be
        raise Exception("Run will overwrite model pickle file, delete or move"
                        " file to continue.")

    submissionsdir = os.path.join(settings.data_dir,"submissions")
    if not os.path.exists(submissionsdir):
        os.mkdir(submissionsdir)

    run_settings['submissions abspath'] = os.path.join(submissionsdir, run_settings['filename'] + ".csv")

    return run_settings


def save_run_settings(run_settings):
    """
    Takes a run_settings dictionary and saves it back where it was loaded from,
    using the path stored in its own dictionary. Return None.
    """
    # don't act on original dictionary
    run_settings = run_settings.copy()
    # store the raw log loss results back in the run settings json
    with open(run_settings['run_settings_path'], 'w') as f:
        # have to remove the settings structure, can't serialise it
        del run_settings['settings']
        json.dump(run_settings, f, separators=(',',':'), indent=4,
                                                    sort_keys=True)
    return None


def format_yaml(run_settings,settings):
    """
    Using the specification from run_settings, will
    substitute in different values into a YAML file
    and write the new YAML file to scratch.
    """
    # open the YAML template
    with open(os.path.join("yaml_templates",run_settings['yaml file'])) as y:
        yaml_string = y.read()
    # sub in the following things for default: settings_path, run_settings_path,
    # final_shape, n_classes, save_path
    hier = neukrill_net.encoding.get_hierarchy(settings)
    hier_group_sizes = {"n_classes_{0}".format(i+1) : n for i, n
                        in enumerate([len(el) for el in hier])}
    run_settings.update(hier_group_sizes)
    run_settings["n_classes"] = len(settings.classes)
    # legacy rename, to make sure it's in there
    run_settings["save_path"] = run_settings['pickle abspath']
    # make new dictionary for substitution
    sub = {}
    sub.update(run_settings)
    sub.update(run_settings['preprocessing'])
    sub.update(run_settings['preprocessing'].get('normalise',{}))
    # time for some crude string parsing
    yaml_string = yaml_string%(sub)
    # write the new yaml to the data directory, in a yaml_settings subdir
    yamldir = os.path.join(settings.data_dir,"yaml_settings")
    if not os.path.exists(yamldir):
        os.mkdir(yamldir)
    yaml_path = os.path.join(yamldir,run_settings["filename"]+
            run_settings['yaml file'].split(".")[0]+".yaml")
    try:
        with open(yaml_path, "w") as f:
            f.write(yaml_string)
    except IOError:
        warnings.warn("Could not write full YAML specification to scratch."
                      " Not required for reproducibility, but can be used "
                      "with Pylearn2 on its own, so may be useful.")
    return yaml_string


def train_test_split(image_fnames, training_set_mode, train_split=0.8, classes=None):
    """
    Perform a stratified split of the image paths stored in a
    image_fnames dictionary supplied.
    Inputs:
        -image_fnames: dictionary of image classes as keys and image paths
    as values.
        -training_set_mode: either "train", "validation" or "test". Will
    split into each based on this.
        -train_split: proportion to split into "train"; remainder split
    equally into "test" and "validation".

    """
    # stratified split of the image paths for train, validation and test
    # iterate over classes, removing some proportion of the elements, in a
    # deterministic way
    test_split = train_split + (1-train_split)/2
    # initialise new variable to store split
    split_fnames = {}
    # assuming train split is some float between 0 and 1, and assign that
    # proportion to train and half of the remaining to test and validation
    if classes is None:
        #raise Warning('You should declare the class names explicitly')
        classes = image_fnames["train"].keys()
        
    for class_label in classes:
        # find where the break should be
        train_break = int(train_split*len(
            image_fnames["train"][class_label]))

        test_break = int(test_split*len(
            image_fnames["train"][class_label]))

        if training_set_mode == "train":
            # then everything up to train_break is what we want
            split_fnames[class_label] \
                    = image_fnames\
                    ["train"][class_label][:train_break]

        elif training_set_mode == "validation":
            # then we want the _first_ half of everything after train_break
            split_fnames[class_label] \
                    = image_fnames \
                    ["train"][class_label][train_break:test_break]

        elif training_set_mode == "test":
            # then we want everything after test_break
            split_fnames[class_label] \
                    = image_fnames \
                    ["train"][class_label][test_break:]

        else:
            raise ValueError("Invalid option for training set mode.")

        # then check it's not empty
        assert len(split_fnames[class_label]) > 0

    return split_fnames


def train_test_split_bool(image_fnames, training_set_mode, train_split=0.8, classes=None):
    """
    Perform a stratified split of the image paths stored in a
    image_fnames dictionary supplied.
    Inputs:
        -image_fnames: dictionary of image classes as keys and image paths
    as values.
        -training_set_mode: either "train", "validation" or "test". Will
    split into each based on this.
        -train_split: proportion to split into "train"; remainder split
    equally into "test" and "validation".
    
    Output:
    ** Returns a boolean of whether each image path in the flattened list
    is included in the split.
    """
    # stratified split of the image paths for train, validation and test
    # iterate over classes, removing some proportion of the elements, in a
    # deterministic way
    test_split = train_split + (1-train_split)/2
    # initialise new variable to store split
    split_bool = np.array([], dtype=bool)
    # assuming train split is some float between 0 and 1, and assign that
    # proportion to train and half of the remaining to test and validation
    if classes is None:
        #raise Warning('You should declare the class names explicitly')
        classes = image_fnames["train"].keys()
        
    for class_label in classes:
        # find where the break should be
        train_break = int(train_split*len(
            image_fnames["train"][class_label]))

        test_break = int(test_split*len(
            image_fnames["train"][class_label]))
        
        len_train = train_break
        len_validation = test_break - train_break
        len_test = len(image_fnames["train"][class_label]) - test_break
        
        if training_set_mode == "train":
            # then everything up to train_break is what we want
            class_split_bool = np.concatenate( (
                                                np.ones(len_train, dtype=bool),
                                                np.zeros(len_validation, dtype=bool),
                                                np.zeros(len_test, dtype=bool)
                                                )  )
            
        elif training_set_mode == "validation":
            # then we want the _first_ half of everything after train_break
            class_split_bool = np.concatenate( (
                                                np.zeros(len_train, dtype=bool),
                                                np.ones(len_validation, dtype=bool),
                                                np.zeros(len_test, dtype=bool)
                                                )  )
            
        elif training_set_mode == "test":
            # then we want everything after test_break
            class_split_bool = np.concatenate( (
                                                np.zeros(len_train, dtype=bool),
                                                np.zeros(len_validation, dtype=bool),
                                                np.ones(len_test, dtype=bool)
                                                )  )
            
        else:
            raise ValueError("Invalid option for training set mode.")
        
        # Check length matches what we expect
        assert len(class_split_bool) == len(image_fnames["train"][class_label])
        # then check it's not empty
        assert sum(class_split_bool) > 0
        
        # Add the class to the full array
        split_bool = np.concatenate((split_bool, class_split_bool))

    return split_bool


def confusion_matrix_from_proba(y_true, y_pred, labels=None):
    """
    Computes a confusion matrix from average of sample probabilites.
    Inputs: y_true - the true labels. Vector
            y_pred - Matrix of predicted probabilities.
                     Each i-th row is the predections for the i-th
                     sample (whose true class is given in y_true[i]
                     Each j-th column is predictions for the sample
                     being a member of the j-th class
    Output: M - Confusion matrix from averaging the probabilities
    """
    y_true = np.array(y_true)
    if labels is None:
        labels = np.union1d(y_true,np.arange(y_pred.shape[1]))
    n_classes = len(labels)
    M = np.zeros((n_classes,n_classes))
    for i in range(n_classes):
        li = (y_true == i)
        M[i,:] = np.mean(y_pred[li,:],0)
    return M


def normalise_cache_range(X_train, X_test=None):
    """
    Normalises cache across dimensions 0 and 1 so they fall
    into [-1,+1] range. Normalises train and test using the
    train data range only.
    Expected input has dimension 0 and 1 as augmentation and image index.
    Each dimension 2 is a different feature.
    """
    # Get the dim0 and dim1 absolute maxima
    X_max = np.amax(np.absolute(X_train.reshape((X_train.shape[0]*X_train.shape[1],X_train.shape[2]))),0)
    # Normalise both arrays
    X_train = X_train / X_max
    if X_test is not None:
        X_test  = X_test / X_max
    
    return X_train, X_test


def save_normalised_cache(train_pkl, test_pkl=None):
    # Load the existing raw data
    X_train = sklearn.externals.joblib.load(train_pkl)
    if test_pkl is None:
        X_test = None
    else:
        X_test  = sklearn.externals.joblib.load(test_pkl)
    # Normalise them both
    X_train, X_test = normalise_cache_range(X_train, X_test)
    # Update names of pickle files
    train_pkl = train_pkl[:-4] + '_ranged' + train_pkl[-4:]
    if test_pkl is not None:
        test_pkl  = test_pkl[:-4] + '_ranged' + test_pkl[-4:]
    # Save to disk
    sklearn.externals.joblib.dump(X_train, train_pkl)
    if test_pkl is not None:
        sklearn.externals.joblib.dump(X_test, test_pkl)
    
