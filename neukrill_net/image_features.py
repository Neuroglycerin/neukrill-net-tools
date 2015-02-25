#!/usr/bin/env python
"""
Module for feature detection and description.

Summary of implemented functions:

- FAST keypoints (todo: look into adaptive thresholds to restrict max no of keypoints)
- ORB keypoints (by default gives maximum 500 keypoints, nothing todo for now)
- BRISK keypoints (todo: look into thresholds)
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


def get_FAST_keypoints(image):
    """
    Detects keypoints using FAST feature detector. 
    (nonmaxSuppression prevents too many keypoints being detected in the same region. This is enabled by default)
    input: image
    output: list of FAST keypoints
    """
    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector()
    # find keypoints
    keyPoints = fast.detect(image, None)

    return keyPoints


def get_ORB_keypoints(image):
    """
    Detects keypoints using ORB feature detector. Maximum no of keypoints returned is 500.
    input: image
    output: list of ORB keypoints
    """
    # Initiate ORB detector
    orb = cv2.ORB()
    # find keypoints
    keyPoints = orb.detect(image, None)

    return keyPoints


def get_BRISK_keypoints(image):
    """
    Detects keypoints using BRISK feature detector.
    input: image
    output: list of BRISK keypoints
    """
    # Initiate BRISK detector
    brisk = cv2.BRISK()
    # find keypoints
    keyPoints = brisk.detect(image, None)

    return keyPoints
