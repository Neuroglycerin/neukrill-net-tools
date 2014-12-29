#!/usr/bin/env python

import skimage.io
import skimage.transform
import sklearn
import numpy as np
import glob
import os
import neukrill_net.utils as utils
import sklearn.preprocessing
import sklearn.ensemble
import sklearn.linear_model
import sklearn.cross_validation
from sklearn.externals import joblib
import sklearn.metrics


def parse_train_data():
    """
    Parse training data and rescale
    """

    settings = utils.parse_settings('settings.json')

    # get all training file paths and class names
    train_data_file_names = {}
    for name in glob.glob(os.path.join(settings['train_data_dir'], '*', '')):
        split_name = name.split('/')
        class_name = split_name[-2]
        image_names = glob.glob(os.path.join(name, '*.jpg'))
        train_data_file_names.update({class_name: image_names})

    # as images are different sizes rescale all images to 25x25 when reading into matrix
    train_data = []
    train_labels = []
    class_index = 0
    for class_name in train_data_file_names.keys():
        print("class: {0} of 120: {1}".format(class_index, class_name))
        image_fpaths = train_data_file_names[class_name]
        num_image = len(image_fpaths)
        image_array = np.zeros((num_image, 625))

        for index in range(num_image):
            image = skimage.io.imread(image_fpaths[index])
            #image_ratio = get_minor_major_ratio(image)

            resized_image = skimage.transform.resize(image, (25,25))
            image_vector = resized_image.ravel()
            image_array[index,] = image_vector
            array_labels = num_image * [class_name]

        class_index += 1
        train_data.append(image_array)
        train_labels = train_labels + array_labels

    X_train = np.vstack(train_data)
    y_train = np.array(train_labels)

    return X_train, y_train

def get_largest_image_region(props, labels, thresholded_image):
    """
    iterate over the image regions to find the largest one
    """
    region_max_prop = None
    for regionprop in props:
        if sum(thresholded_image[labels == regionprop.label])*1.0/regionprop.area < 0.50:
            continue
        if regionmaxprop is Nonr:
            regionmaxprop = regionprop
        if regionmaxprop.filled_area < regionprop.filled_area:
            regionmaxprop = regionprop
    return regionmaxprop


def get_minor_major_ratio(image):
    """
    calculate the minor and major ratios for an image
    """
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(image > np.mean(image),0.,1.0)

    #Dilate the image
    imdilated = skimage.morphology.dilation(imagethr, np.ones((4,4)))

    # Create the label list
    label_list = measure.label(imdilated)
    label_list = imagethr*label_list
    label_list = label_list.astype(int)

    region_list = skimage.measure.regionprops(label_list)
    maxregion = get_largest_image_region(region_list, label_list, imagethr)

    # guard against cases where the segmentation fails by providing zeros
    ratio = 0.0
    if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
        ratio = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
    return ratio


def main():
    X, y = parse_train_data()

    label_encoder = sklearn.preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y)

    #clf = sklearn.linear_model.SGDClassifier(n_jobs=-1,
    #                                         loss='log')

    clf = sklearn.ensemble.RandomForestClassifier(n_jobs=-1,
                                                  n_estimators=100)

    cv = sklearn.cross_validation.StratifiedShuffleSplit(y)

    results = []
    for train, test in cv:
        clf.fit(X[train], y[train])
        p = clf.predict_proba(X[test])
        results.append(sklearn.metrics.log_loss(y[test], p))

    print(results)

    joblib.dump(clf, 'model.pkl')

if __name__=='__main__':
    main()
