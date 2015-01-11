#!/usr/bin/env python
"""
Unit tests for utility functions
"""
import os
import glob
import shutil
import io
import numpy as np
from neukrill_net.tests.base import BaseTestCase
import neukrill_net.utils as utils

class TestSettings(BaseTestCase):
    """
    Unit tests for settings class
    """

    def setUp(self):
        """
        Set up tests by ensuring that the stringIO object used in testing
        returns the same output as a real json file
        """
        self.data_dir = "TestSettingsParserDir"
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
            os.mkdir(os.path.join(self.data_dir, 'test'))
            os.mkdir(os.path.join(self.data_dir, 'train'))

        self.check_settings_file = os.path.join(self.test_dir,
                                                'check_settings_file.json')

    def test_init_with_correct_file_input(self):
        """
        Ensure a valid dict is created from a correct json during init
        """
        settings = utils.Settings(self.check_settings_file)
        self.assertIs(settings.user_input.__class__, dict)
        self.assertTrue(len(settings.user_input) > 0)

    def test_init_with_correct_stringIO(self):
        """
        Ensure a valid dict is created from a correct json io.String during init
        """
        string_settings = io.StringIO('{"data_dir": ["TestSettingsParserDir"]}')
        settings = utils.Settings(string_settings)
        self.assertIs(settings.user_input.__class__, dict)
        self.assertTrue(len(settings.user_input) > 0)

    def test_error_if_file_does_not_exist(self):
        """
        Ensure an IOError is thrown if the file doesn't exist
        """
        with self.assertRaises(ValueError):
            utils.Settings('fake_file')

    def test_error_if_required_missing(self):
        """
        Ensure error is thrown if a required setting is omitted
        """
        setting_string_without_required = io.StringIO('{"foo": 5, "bar": "duck"}')
        with self.assertRaises(ValueError):
            utils.Settings(setting_string_without_required)

    def test_resolves_to_correct_dir(self):
        """
        Make sure settings parser resolves to dir containing test and train dirs
        """
        settings_string_with_2_dirs = io.StringIO('{"data_dir": ["fake", "TestSettingsParserDir"]}')
        settings = utils.Settings(settings_string_with_2_dirs)
        self.assertEqual(settings.data_dir, os.path.abspath(self.data_dir))

    def check_default_values_during_init(self):
        """
        Make sure default values are set for r_seed
        """
        settings = utils.Settings(self.check_settings_file)
        classes = ("acantharia_protist",
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
                   "unknown_unclassified")

        self.assertIs(settings.random_seed, 42)
        self.assertIs(settings.classes, classes)

    def tearDown(self):
        """
        Remove data dir
        """
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)


class TestLoadData(BaseTestCase):
    """
    Test load data util function
    """

    def setUp(self):
        """
        Initialise image_fname_dict and classes
        """
        # unecessary but just to make it clear these values are coming from
        # superclass
        self.image_fname_dict = self.image_fname_dict
        self.classes = self.classes

        self.processing = lambda image: np.zeros((10,10))

    def test_load_train_without_processing(self):
        """
        Check load_data fails stacking training data without a processing step
        """
        with self.assertRaises(ValueError):
            _, _ = utils.load_data(self.image_fname_dict,
                                   classes=self.classes)

    def test_loading_train_data_with_processing(self):
        """
        Ensure load_data with training data returns the correct data
        """
        data, labels = utils.load_data(self.image_fname_dict,
                                       classes=self.classes,
                                       processing=self.processing)

        self.assertIs(len(labels), 10)
        self.assertEqual(['acantharia_protist'] * 3 + \
                         ['acantharia_protist_halo'] * 2 + \
                         ['artifacts_edge'] * 4 + \
                         ['fecal_pellet'], list(labels))
        self.assertEqual(data.shape, (10, 100))

    def test_load_train_data_name_correspondence_is_correct(self):
        """
        Ensure the correspondence of labels to data is maintained
        on load
        """
        single_val_processing = lambda images: images.min()

        data, labels = utils.load_data(self.image_fname_dict,
                                       classes=self.classes,
                                       processing=single_val_processing)

        self.assertIs(len(labels), 10)
        self.assertEqual(['acantharia_protist'] * 3 + \
                         ['acantharia_protist_halo'] * 2 + \
                         ['artifacts_edge'] * 4 + \
                         ['fecal_pellet'], list(labels))
        self.assertEqual(data.shape, (10, 1))
        self.assertEqual([[int(x[0])] for x in data], [[51], [73], [65], [35],
                                                       [37], [202], [0], [0],
                                                       [0], [158]])

    def test_load_test_fails_without_processing(self):
        """
        Make sure load_data fails to stack training data without processing
        """
        with self.assertRaises(ValueError):
            _, _ = utils.load_data(self.image_fname_dict)

    def test_loading_test_data_with_processing(self):
        """
        Check whether data and names are correct when loading test data
        with dummy zeros((10,10)) processing
        """
        data, names = utils.load_data(self.image_fname_dict,
                                      processing=self.processing)

        self.assertEqual(['136177.jpg', '81949.jpg', '27712.jpg'], names)
        self.assertEqual(data.shape, (3, 100))

    def test_load_test_data_name_correspondence_is_correct(self):
        """
        Make sure the names match up to the correct row in the data for test
        data
        """
        single_val_processing = lambda images: images.min()
        data, names = utils.load_data(self.image_fname_dict,
                                      processing=single_val_processing)

        self.assertEqual(['136177.jpg', '81949.jpg', '27712.jpg'], names)
        self.assertIs(int(data[0][0]), 63)
        self.assertIs(int(data[1][0]), 46)
        self.assertIs(int(data[2][0]), 5)

