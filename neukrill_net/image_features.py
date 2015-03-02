#!/usr/bin/env python
"""
Module for feature detection and description.

Summary of implemented functions:

- FAST keypoints
- ORB keypoints
- MSER keypoints
- BRISK keypoints

We can now specify the desired no of keypoints. Fitness is based on 'response' value of keypoints.

TODO: -look into adaptive thresholds to generate keypoints if no points are detected with default parameters.
      -make sure if varying patchSize for ORB, descriptor uses same parameters! (DONE - we use **kwargs)
"""

import cv2
import numpy as np

"""
####
Keypoint Detection and description
####
"""

def sort_keypoints_by_response_and_get_n_best(keypoint_list, n=500):
    """
    Sorts keypoint list by "response" field and returns the first n best keypoints. 
    If the length of the list is smaller than n, than return the whole list.
    input: keypoint_list - list of keypoints
           n - no of keypoints to be returned
    """
    sortedList = sorted(keypoint_list, key=lambda x: x.response, reverse=True)

    bestNKeypoints = []

    if (len(sortedList) > n):
        bestNKeypoints = sortedList[:n]
    else:
        bestNKeypoints = sortedList

    return bestNKeypoints



def get_FAST_keypoints(image, n=500):
    """
    Detects keypoints using FAST feature detector. 
    (nonmaxSuppression prevents too many keypoints being detected in the same region. This is enabled by default)
    input:  image
            n - max number of returned keypoints (default 500)
    output: list of FAST keypoints
    """
    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector()

    # find keypoints
    keyPoints = fast.detect(image, None)

    # sort by response and return best n keypoints
    keyPoints = sort_keypoints_by_response_and_get_n_best(keyPoints, n)

    return image, keyPoints, {}


def get_ORB_keypoints(image, n=500, **kwargs):
    """
    Detects keypoints using ORB feature detector. Maximum no of keypoints returned is 500.
    input: image
           n - max number of returned keypoints (default 500)
    output: list of ORB keypoints
    """
    # blur using a Gaussian kernel
    #image = cv2.GaussianBlur(image,(3,3),0)
    #image = cv2.bilateralFilter(image,5,75,75)

    keyPoints = []
    thePatchSize = 31
    # find keypoints; if none found, decrease patchSize
    while ( len(keyPoints) == 0 ):
        # Initiate ORB detector
        orb = cv2.ORB(nfeatures = n, edgeThreshold = 0, patchSize = thePatchSize)
        keyPoints = orb.detect(image, None)
        if thePatchSize <= 3:
            break
        thePatchSize -= 2
        

    # sort by response and return best n keypoints
    # already scored by "scoreType" HARRIS_SCORE?
    keyPoints = sort_keypoints_by_response_and_get_n_best(keyPoints, n)

    return image, keyPoints, {"patchSize" : thePatchSize}


def get_ORB_descriptions(image, keyPoints, patchSize=9, **kwargs):
    """
    Computes ORB descriptions for given keypoints.
    input:  image (that was returned with the keypoints!)
            keyPoints - detected keypoints
            **kwargs = detection arguments
    output: list of descriptions for given keypoints
    """
    orb = cv2.ORB(edgeThreshold = 0, patchSize = patchSize)
    # gets keypoints and descriptions
    kp, descriptions = orb.compute(image, keyPoints)

    if descriptions is None:
        descriptions = np.array([])

    return descriptions


def get_BRISK_keypoints(image, n=500):
    """
    Detects keypoints using BRISK feature detector.
    input:  image
            n - max no of returned keypoints (default 500)
    output: list of BRISK keypoints
    """
    # Initiate BRISK detector
    brisk = cv2.BRISK()

    # find keypoints
    keyPoints = brisk.detect(image, None)

    # sort by response and return best n keypoints
    keyPoints = sort_keypoints_by_response_and_get_n_best(keyPoints, n)

    return image, keyPoints, {}


def get_BRISK_descriptions(image, keyPoints, **kwargs):
    """
    Computes BRISK descriptions for given keypoints.
    input:  image (that was returned with the keypoints!)
            keyPoints - detected keypoints
            **kwargs = detection arguments
    output: list of descriptions for given keypoints
    """
    brisk = cv2.BRISK()
    # gets keypoints and descriptions
    kp, descriptions = brisk.compute(image, keyPoints)

    if descriptions is None:
        descriptions = np.array([])

    return descriptions


def get_MSER_keypoints(image, n=500):
    """
    Detects keypoints using MSER detector. 
    input:  image
            n - max number of returned keypoints (default 500)
    output: list of MSER keypoints
    """
    # initiate MSER detector
    mser = cv2.FeatureDetector_create('MSER')

    # find keypoints
    keyPoints = mser.detect(image)

    # sort by response and return best n keypoints
    keyPoints = sort_keypoints_by_response_and_get_n_best(keyPoints, n)

    return image, keyPoints, {}


"""
####
Shape features:
- moments
- Hu moments
####
"""


def get_shape_moments(image):
    """
    Returns dictionary of moments:
    http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#moments
    input: image
    output: dictionary of moments (see OpenCV documentation)
    """
    #im = copy.deepcopy(rawdata[0])
    ret,thresh = cv2.threshold(image,254,255,cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Find the index of the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    M = cv2.moments(cnt)

    #cv2.drawContours(image, cnt, -1, 2)
    #cv2.imwrite('blabla.png',image)
    return M


def get_shape_HuMoments(moments):
    """
    Returns Hu moments (invariants to the image scale, rotation, and reflection).
    http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#humoments

    input: dictionary of moments computed with get_shape_moments
    output: numpy array of the 6 moments (7th one is removed as the sign is changed by reflection)
    """
    huMoments = cv2.HuMoments(moments).flatten()
    
    # take log as advised by internetz
    huMoments = -np.sign(huMoments) * np.log10(np.abs(huMoments))

    # remove last moment as it is dependent on reflection
    huMoments = huMoments[:6]

    return huMoments



