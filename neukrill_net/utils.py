#!/usr/bin/env python
"""
Utility functions for the neukrill-net classifier
"""
import glob
import io
import json
import os
import skimage.io
import skimage.transform

class Settings:
    """
    A class to handle all the parsing of the settings.json as well as
    checking the values and providing defaults
    """
    def __init__(self, settings_file):
        """
        Initialise settings object
        """
        required_settings = ['data_dir']

        # read the provided settings json (or stringIO)
        parsed_settings = self.parse_settings(settings_file)

        # check all required options have been set
        self.user_input = self.check_mandatory_fields(parsed_settings,
                                                      required_settings)

        # check the data dir options and find the correct one
        self.data_dir = self.check_data_dir(self.user_input['data_dir'])

        # check user input for random seed and if absent set it
        self.random_seed = self.user_input.get('r_seed', 42)

        # check user defined classes if absent set it
        self.classes = self.user_input.get('classes',
            ("acantharia_protist",
             "acantharia_protist_big_center",
             "acantharia_protist_halo",
             "amphipods",
             "appendicularian_fritillaridae",
             "appendicularian_slight_curve",
             "appendicularian_s_shape",
             "appendicularian_straight",
             "artifacts",
             "artifacts_edge",
             "chaetognath_non_sagitta",
             "chaetognath_other",
             "chaetognath_sagitta",
             "chordate_type1",
             "copepod_calanoid",
             "copepod_calanoid_eggs",
             "copepod_calanoid_eucalanus",
             "copepod_calanoid_flatheads",
             "copepod_calanoid_frillyAntennae",
             "copepod_calanoid_large",
             "copepod_calanoid_large_side_antennatucked",
             "copepod_calanoid_octomoms",
             "copepod_calanoid_small_longantennae",
             "copepod_cyclopoid_copilia",
             "copepod_cyclopoid_oithona",
             "copepod_cyclopoid_oithona_eggs",
             "copepod_other",
             "crustacean_other",
             "ctenophore_cestid",
             "ctenophore_cydippid_no_tentacles",
             "ctenophore_cydippid_tentacles",
             "ctenophore_lobate",
             "decapods",
             "detritus_blob",
             "detritus_filamentous",
             "detritus_other",
             "diatom_chain_string",
             "diatom_chain_tube",
             "echinoderm_larva_pluteus_brittlestar",
             "echinoderm_larva_pluteus_early",
             "echinoderm_larva_pluteus_typeC",
             "echinoderm_larva_pluteus_urchin",
             "echinoderm_larva_seastar_bipinnaria",
             "echinoderm_larva_seastar_brachiolaria",
             "echinoderm_seacucumber_auricularia_larva",
             "echinopluteus",
             "ephyra",
             "euphausiids",
             "euphausiids_young",
             "fecal_pellet",
             "fish_larvae_deep_body",
             "fish_larvae_leptocephali",
             "fish_larvae_medium_body",
             "fish_larvae_myctophids",
             "fish_larvae_thin_body",
             "fish_larvae_very_thin_body",
             "heteropod",
             "hydromedusae_aglaura",
             "hydromedusae_bell_and_tentacles",
             "hydromedusae_h15",
             "hydromedusae_haliscera",
             "hydromedusae_haliscera_small_sideview",
             "hydromedusae_liriope",
             "hydromedusae_narco_dark",
             "hydromedusae_narcomedusae",
             "hydromedusae_narco_young",
             "hydromedusae_other",
             "hydromedusae_partial_dark",
             "hydromedusae_shapeA",
             "hydromedusae_shapeA_sideview_small",
             "hydromedusae_shapeB",
             "hydromedusae_sideview_big",
             "hydromedusae_solmaris",
             "hydromedusae_solmundella",
             "hydromedusae_typeD",
             "hydromedusae_typeD_bell_and_tentacles",
             "hydromedusae_typeE",
             "hydromedusae_typeF",
             "invertebrate_larvae_other_A",
             "invertebrate_larvae_other_B",
             "jellies_tentacles",
             "polychaete",
             "protist_dark_center",
             "protist_fuzzy_olive",
             "protist_noctiluca",
             "protist_other",
             "protist_star",
             "pteropod_butterfly",
             "pteropod_theco_dev_seq",
             "pteropod_triangle",
             "radiolarian_chain",
             "radiolarian_colony",
             "shrimp_caridean",
             "shrimp-like_other",
             "shrimp_sergestidae",
             "shrimp_zoea",
             "siphonophore_calycophoran_abylidae",
             "siphonophore_calycophoran_rocketship_adult",
             "siphonophore_calycophoran_rocketship_young",
             "siphonophore_calycophoran_sphaeronectes",
             "siphonophore_calycophoran_sphaeronectes_stem",
             "siphonophore_calycophoran_sphaeronectes_young",
             "siphonophore_other_parts",
             "siphonophore_partial",
             "siphonophore_physonect",
             "siphonophore_physonect_young",
             "stomatopod",
             "tornaria_acorn_worm_larvae",
             "trichodesmium_bowtie",
             "trichodesmium_multiple",
             "trichodesmium_puff",
             "trichodesmium_tuft",
             "trochophore_larvae",
             "tunicate_doliolid",
             "tunicate_doliolid_nurse",
             "tunicate_partial",
             "tunicate_salp",
             "tunicate_salp_chains",
             "unknown_blobs_and_smudges",
             "unknown_sticks",
             "unknown_unclassified"))

        # a way to encode the superclasses if we need to but don't want to complete
        # just now as it is very monotonous to copy everything
        # self.super_classes = {'FISH': ('fish classes'),
        #                       'DIATOMS': ('diatom classes'),
        #                       'TRICHODESMIUM': ('you get the idea')
        #                       'PROTISTS': ('protists_classes')
        #                       'NO_SUPER_CLASS': ('unclassified')}

        self._image_fnames = {}

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


    def check_mandatory_fields(self, parsed_user_input, required_settings):
        """
        Ensure all the mandatory settings are present
        """
        for entry in required_settings:
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
            test_fnames = tuple(glob.glob(os.path.join(self.data_dir,
                                                       'test',
                                                       '*.jpg')))

            # check there are the correct number of images
            num_test_images = len(test_fnames)
            if num_test_images != 130400:
                raise ValueError('Wrong number of test images found: {0}'
                                 ' instead of 130400'.format(num_test_images))

            train_fnames = {}

            for name in glob.glob(os.path.join(self.data_dir,
                                               'train',
                                               '*',
                                               '')):
                split_name = name.split('/')
                class_name = split_name[-2]
                image_names = glob.glob(os.path.join(name, '*.jpg'))
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
        print("Warning: no processing applied, it will \
        not be possible to stack these images due to \
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
