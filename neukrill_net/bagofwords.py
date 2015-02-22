#!/usr/bin/env python
"""
Bag Of visual Words classifier
"""

from __future__ import division

import numpy as np
import cv2
import sklearn.cluster

from neukrill_net import image_attributes


def attributes_wrapper(attributes_settings):
    """
    Builds a function which, given an image, spits out
    a vector of scalars each corresponding to the
    attributes of the image which were requested in the
    settings provided to this function.
    """
    # Make a list of functions corresponding to each of the
    # attributes mentioned in the settings
    funcvec = []
    for attrfuncname in attributes_settings:
        # From the attributes module, lookup the function
        # bearing the target name 
        funcvec += [getattr(image_attributes, attrfuncname)]
    
    # Make a function which applies all the functions to the image
    # returning them in a list
    # NB: must be a numpy array so we can "ravel" it
    return lambda image: np.asarray([f(image) for f in funcvec])


class Bow:
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
        self.detector = cv2.ORB_create(
            nfeatures = options['n_features_max'],
            patchSize = options['patch_size'],
            edgeThreshold = 0)
        
        
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
        
    
    def describeImage(self, img):
        """
        Get all the keypoints in an image and describe their local patches.
        
        Input: img - An image as a 2D numpy array
        
        Output: des - A numpy array sized (num_keypoints, 32)
                      where num_keypoints is the number of keypoints,
                      which varies between images.
        """
        _, des = self.detector.detectAndCompute(img, None)
        return des
    
    def build_vocabulary(self, images):
        """
        Given a set of training images, define what constitutes "words" by
        clustering the descriptors of all keypoint patches.
        
        Input: images - Training image set formatted as a list of numpy arrays
        """
        if self.verbose:
            print('Describing the keypoints of {} images'.format(len(images)))
        
        # For each image, get all keypoint descriptions
        X = [self.describeImage(img) for img in images]
        
        # Remove empty descriptions from images without any keypoints
        X = [x for x in X if x is not None]
        
        # Flatten so we have all keypoints from all images on top of each other
        X = np.vstack(X)
        
        # Tell whatever clustering algorithm instance we have to fit
        # itself to this data
        if self.verbose:
            print('Clustering patch descriptors to form vocabulary')
        
        self.cluster.fit(X)
        
    def compute_image_bow(self, img):
        """
        Computes the bag of words associated with an image.
        Computes descriptions of every keypoint in an image
        and assigns them to a word using the pretrained vocabulary
        in the self.cluster classifier.
        Then returns a histogram of the number of keypoints which
        fall into each of the classes.
        NB: This method serves as the wrapper function for BOW.
        
        Input:  img - an image as a numpy array
        
        Output: N - histogram of word occurances as a numpy array
                    size is (n_clusters,)
                    where n_clusters is the number of words in the
                    vocabulary.
        """
        # Get the descriptions of every keypoint in the image
        X = self.describeImage(img)
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
            N /= np.sum(N)
        return N
        
    @property 
    def n_clusters(self):
        return self.cluster.cluster_centers_.shape[0]

