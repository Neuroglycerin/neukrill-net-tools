#!/usr/bin/env python

import neukrill_net.utils as utils
import csv
import pickle
from sklearn.externals import joblib
import skimage.io
import skimage.transform
import numpy as np
import glob
import os


def parse_test_data():
    """
    Parse the test dataset and resize to 25x25
    """

    settings = utils.parse_settings('settings.json')

    test_data_paths = glob.glob(os.path.join(settings['test_data_dir'], '*.jpg'))
    test_data = {image.split('/')[-1]: image for image in test_data_paths}

    image_name_list = []
    num_images = len(test_data)
    image_index = 0
    image_array = np.zeros((num_images, 625))

    for image_name in test_data.keys():
        image_name_list.append(image_name)

        image = skimage.io.imread(test_data[image_name])
        resized_image = skimage.transform.resize(image, (25,25))
        image_vector = resized_image.ravel()

        image_array[image_index, ] = image_vector

        image_index += 1
        print("image: {0} of 130400: {1}".format(image_index, image_name))


    X_test = image_array
    image_names = np.array(image_name_list)
    return X_test, image_names


def main():
    with open('class_labels.pkl', 'rb') as labels_fh:
        class_labels = pickle.load(labels_fh)


    X, names = parse_test_data()
    clf = joblib.load('model.pkl')
    p = clf.predict_proba(X)

    with open('submission.csv', 'w') as csv_out:
        out_writer = csv.writer(csv_out, delimiter=',')
        out_writer.writerow(['image'] + class_labels)
        for index in range(len(names)):
            out_writer.writerow([names[index]] + list(p[index,]))

if __name__ == '__main__':
    main()

