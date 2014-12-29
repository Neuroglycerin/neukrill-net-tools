#!/usr/bin/env python

import skimage.io
import skimage.transform
import sklearn
import numpy as np
import glob
import os
import neukrill_net.utils as utils

def main():

    settings = utils.parse_settings('settings.json')

    # get all training file paths and class names
    train_data_file_names = {}
    for name in glob.glob(os.path.join(settings['train_data_dir'], '*', '')):
        split_name = name.split('/')
        class_name = split_name[-2]
        image_names = glob.glob(os.path.join(name, '*.jpg'))
        train_data_file_names.update({class_name: image_names})

    # as images are different sizes rescale all images to 25x25 when reading into matrix
    train_data = {}
    class_index = 0
    for class_name in train_data_file_names.keys():
        print("class: {0} of 120: {1}".format(class_index, class_name))
        image_fpaths = train_data_file_names[class_name]
        num_image = len(image_fpaths)
        image_array = np.zeros((num_image, 625))

        for index in range(num_image):
            image = skimage.io.imread(image_fpaths[index])
            resized_image = skimage.transform.resize(image, (25,25))
            image_vector = resized_image.ravel()
            image_array[index,] = image_vector

        class_index += 1
        train_data.update({class_name: image_array})


if __name__=='__main__':
    main()
