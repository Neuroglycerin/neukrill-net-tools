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

