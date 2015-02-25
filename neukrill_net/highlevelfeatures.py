#!/usr/bin/env python
"""
Module for feature extractor object classes and classifier functions.
"""

from __future__ import division

import numpy as np
import cv2
import sklearn.cluster
import six

from neukrill_net import image_attributes

def loadimage(obj):
    """
    Loads an image if a path is given
    Input : an image or a path
    Output: the image given or loaded from path
    """
    if isinstance(obj, np.ndarray):
        # Input is a loaded image
        return obj
    elif isinstance(obj, basestring) or isinstance(obj, six.string_types):
        # Input is a string. Assume it is a path.
        # Try to load the image from this path
        return skimage.io.imread(obj)
    else:
        # No duck typing for this function, so we will give an error
        # if we don't see the right kind of input.
        raise ValueError("Wrong kind of object given. Not a numpy array or a string.")


class HighLevelFeatureBase:
    """
    Base class for high level features
    """
    def __init__(self, preprocessing_func=None, augment_func=None):
        """
        Initialise the preprocessing and augmentation functions.
        """
        if not preprocessing_func is None:
            self.preprocess_image = preprocessing_func
            
        if not augment_func is None:
            self.augment_image = augment_func
        
            
    def fit(self, images):
        """
        Fit the feature to a training set.
        Some subclasses will support this, but not all.
        Input : a list of images or image paths
        Output: None
        """
        raise NotImplementedError
        
        
    def preprocess_image(self, image):
        """
        Preprocessing function
        Input : image - a 2D numpy array of an image
        Output: image - a 2D numpy array of the processed image
                        the output may not be the same size as the input
        """
        # By default, the preprocessing function does nothing to the 
        # input image
        return image
        
        
    def augment_image(self, image):
        """
        Augmentation method
        Takes an image and returns a list of augmented versions of it.
        The number of augmentations should be the same for every input.
        Input : image  - a 2D numpy array of an image
        Output: images - a list of numpy arrays of images
        """
        # By default, we do not augment and return the input image
        # on its own in a list
        return [image]
        
        
    def extractfeatures_image(self, image):
        """
        Extracts a feature vector from the image
        Input : image - a 2D numpy array of an image
        Output: features - a 1D numpy array of features extracted from the image.
        """
        # Subclasses will need to overwrite this function
        raise NotImplementedError
        
        
    def extractfeatures(self, images):
        """
        Extract features from a set of images.
        
        Input : images - a list of images as 2D numpy arrays
                            OR
                         a list of paths of images
        
        Output: features - a 3D numpy array of features extracted from
                           the images.
                           The size of the numpy array is
                           (num_augmentations, num_images, feature_length)
        
        NOTE: Subclasses should not modify this function!
        """
        # Probe object's other methods to see how big my output needs to be
        first_image = loadimage(images[0])
        # How many augmentations do we get from each image?
        num_augmentations = len(self.augment_image(first_image))
        # How many elements are in the feature vector from each image?
        num_feature_elements = self.extractfeatures_image(
                                    self.preprocess_image(first_image)
                                    ).size
        
        # Initialise
        X = np.zeros((num_augmentations, len(images), num_feature_elements))
        
        # Loop over list of images
        for image_index, image in enumerate(images):
            # Load the image if necessary
            image = loadimage(image)
            # Augment the image, giving a list of copies
            augmented_list = self.augment_image(image)
            # Loop over all the augmented copies
            for augment_index, augmented_image in enumerate(augmented_list):
                # Preprocess the image
                augmented_image = self.preprocess_image(augmented_image)
                # Extract features and put them into the array
                X[augment_index, image_index, :] = np.ravel(
                                                    self.extractfeatures_image(augmented_image)
                                                    )
        
        # Return the completed arary
        return X



class BasicAttributes(HighLevelFeatureBase):
    """
    Get generic, basic, high level attributes from the image
    """
    def __init__(self, attributes_list, **kwargs):
        # Set the feature extractor function to provide the target list
        # of attributes
        self.extractfeatures_image = attributes_wrapper(attributes_list)
        
        # Call parent's initialization routine
        super(self.__class__,self).__init__(**kwargs)
        
        
    @staticmethod
    def attributes_wrapper(attributes_list):
        """
        Generates a function which, given an image, spits out
        a vector of scalars each corresponding to the
        attributes of the image which were requested in the
        settings provided to this function.
        
        Input : a list of strings of image attributes
        Output: a function which maps an image to a numpy vector
        """
        # Make a list of functions corresponding to each of the
        # attributes mentioned in the settings
        funcvec = []
        for attrfuncname in attributes_list:
            # From the attributes module, lookup the function
            # bearing the target name 
            funcvec += [getattr(image_attributes, attrfuncname)]
        
        # Make a function which applies all the functions to the image
        # returning them in a list
        # NB: must be a numpy array so we can "ravel" it
        return lambda image: np.asarray([f(image) for f in funcvec]) 



class BagOfWords(HighLevelFeatureBase):
    """
    A class for traditional Bag Of visual Words using a histogram of
    the clusters (words) within which local features fall.
    """
    
    def __init__(self, verbose=False, normalise_hist=False, **options):
        """Initialisation"""
        
        # Set parameters
        self.verbose = verbose
        self.normalise_hist = normalise_hist
        
        # Make a keypoint detector and patch describer
        # We use ORB because it is free, whereas SIFT and SURF are not
        # NB: the edge threshold controls how much of the image must be ignored from
        #     each edge. This needs to be set to something very low,
        #     otherwise it will ignore all of the smaller (21x30) images.
        self.detector = cv2.ORB(
                                nfeatures = options['n_features_max'],
                                patchSize = options['patch_size'],
                                edgeThreshold = 0
                                )
        
        
        # Add whichever cluster object we should use
        # Needs to be one which supports both fit and predict
        if options['clusteralgo'].lower()=='kmeans':
            # K-means clustering algorithm
            self.cluster = sklearn.cluster.KMeans(
                                n_clusters = options['n_clusters'],
                                n_init = 10,
                                max_iter = 300,
                                random_state = options['random_seed'],
                                n_jobs = 1
                                )
            
        elif options['clusteralgo'].lower()=='meanshift':
            # Mean-shift clustering algorithm
            sklearn.cluster.MeanShift(
                                bandwidth = None,
                                seeds = None,
                                bin_seeding = False,
                                min_bin_freq = 1,
                                cluster_all = True
                                )
            
        elif options['clusteralgo'].lower()=='affinity':
            # Affinity Propagation algorithm
            sklearn.cluster.AffinityPropagation(
                                damping = 0.5,
                                max_iter = 200,
                                convergence_iter = 15
                                )
            
        else:
            # Didn't match, so raise error
            raise ValueError('Unrecognised clustering algorithm "{}"'.format(options['clusteralgo']))
        
    
    def describeImage(self, image):
        """
        Get all the keypoints in an image and describe their local patches.
        
        Input: image - An image as a 2D numpy array
        
        Output: des - A numpy array sized (num_keypoints, 32)
                      where num_keypoints is the number of keypoints,
                      which varies between images.
        """
        _, des = self.detector.detectAndCompute(image, None)
        return des
    
    
    def fit(self, images):
        """
        Given a set of training images, define what constitutes "words" by
        clustering the descriptors of all keypoint patches.
        
        Input: images - Training image set formatted as a list of numpy arrays
        """
        if self.verbose:
            print('Describing the keypoints of {} images'.format(len(images)))
        
        # For each image, get all keypoint descriptions
        X = [self.describeImage(loadimage(image)) for image in images]
        
        # Remove empty descriptions from images without any keypoints
        X = [x for x in X if x is not None]
        
        # Flatten so we have all keypoints from all images on top of each other
        X = np.vstack(X)
        
        # Tell whatever clustering algorithm instance we have to fit
        # itself to this data
        if self.verbose:
            print('Clustering patch descriptors to form vocabulary')
        
        self.cluster.fit(X)
        
        
    def extractfeatures_image(self, image):
        """
        Computes the bag of words associated with an image.
        Computes descriptions of every keypoint in an image
        and assigns them to a word using the pretrained vocabulary
        in the self.cluster classifier.
        Then returns a histogram of the number of keypoints which
        fall into each of the classes.
        
        Input:  image - an image as a numpy array
        
        Output: N - histogram of word occurances as a numpy array
                    size is (n_clusters,)
                    where n_clusters is the number of words in the
                    vocabulary.
        """
        # Get the descriptions of every keypoint in the image
        X = self.describeImage(image)
        if X is None:
            # No keypoints, so occurances of any words 
            return np.zeros(self.n_clusters)
        # Predict the class of every keypoint
        y = self.cluster.predict(X)
        # Make a histogram of the classes
        N = np.bincount(y, minlength=self.n_clusters)
        # Normalise the histogram so it is proportion of words
        # relative to total number of words instead of absolute
        # number of occurances of each word
        if self.normalise_hist:
            N = N / np.sum(N)
        return N
        
        
    @property 
    def n_clusters(self):
        return self.cluster.cluster_centers_.shape[0]

