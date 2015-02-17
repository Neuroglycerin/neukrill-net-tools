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
import numpy as np
import sklearn.preprocessing

class DensePNGDataset(pylearn2.datasets.DenseDesignMatrix):
    """
    A class intended to load images from a directory, apply some
    processing and initialise a pylearn2 DenseDesignmatrix model.

    Parameters
    ----------
    settings_path : settings.json file path used to find images and control
        how they are loaded and processed.
    """
    def __init__(self,settings_path="settings.json",
            run_settings"run_settings/default.json",train_or_predict="train"):
        # parse the settings file
        self.settings = neukrill_net.utils.Settings(settings_path)
        # get the processing settings
        self.run_settings = run_settings
        processing_settings = self.run_settings["preprocessing"]
        # get a processing function from this
        processing = neukrill_net.augment.augmentation_wrapper(
                                                        processing_settings)
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
            print(class_label)
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
        # make sure y is an array
        y = np.array(y)
        # count the y labels
        N_y_labels = len(list(set(y)))
        # need to encode labels numerically
        self.label_encoder = sklearn.preprocessing.LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        # now run inherited initialisation
        super(self.__class__,self).__init__(topo_view=X,y=y,y_labels=N_y_labels)
