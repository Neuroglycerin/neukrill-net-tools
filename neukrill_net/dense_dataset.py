"""
This is intended to be a simple inheritance of the DenseDesignMatrix class
of pylearn2. The initialisation has been modified to use a settings file
to load the images and process them before running the standard initialisation.
"""
__authors__ = "Gavin Gray"
__copyright__ = "Copyright 2015 - University of Edinburgh"
__credits__ = ["Gavin Gray"]
__license__ = "3-clause BSD"
__maintainer__ = "Gavin Gray"
__email__ = "gavingray1729@gmail.com"

import skimage.io
import neukrill_net.utils
import neukrill_net.augment
import pylearn2.datasets

class DensePNGDataset(pylearn2.datasets.DenseDesignMatrix):
    """
    A class intended to load images from a directory, apply some
    processing and initialise a pylearn2 DenseDesignmatrix model.

    Parameters
    ----------
    settings : settings.json file used to find images and control
        how they are loaded and processed.
    """
    def __init__(self,settings_path):
        # parse the settings file
        self.settings = neukrill_net.utils.Settings(settings_path)
        # get the processing settings
        processing_settings = self.settings.user_input['processing']
        # get a processing function from this
        processing = neukrill_net.augment.augmentation_wrapper(
                                                        processing_settings)
        # initialise y vector
        y = []
        # initialise list of arrays (which will contain images
        X = []
        # load the images in image_fpaths, iterating and keeping track of class
        for class_label in self.settings.classes:
            for image_path in self.settings.images_fpaths[class_label]:
                # load the image as numpy array
                image = skimage.io.imread(image_path)
                # apply processing function (get back multiple images)
                images = processing(image)
                # for each image store a class label
                y += [class_label]*len(images)

