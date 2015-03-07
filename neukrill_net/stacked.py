#!/usr/bin/env python
"""
Module hierarchically stacked classifiers
"""

from __future__ import division

import sklearn.cross_validation

class StackedClassifier():
    """
    Splits data so it can fit an inner and outer classifier
    """
    def __init__(self, hlf, clf, inner_prop=0.25, random_state=42, ensure_split=False):
        """
        Initialise
        """
        self.hlf = hlf
        self.clf = clf
        self._inner_prop = inner_prop
        self.random_state
        self._ensure_split = ensure_split
        
        
    def fit(self, X, y):
        """
        Fit a high level feature on one split of the training data, and the outer
        classifier on the rest of the training data.
        """
        if not self._ensure_split and not getattr(self.hlf, '_needs_fitting', True):
            # Inner high level features do not need fitting
            # Ignore the inner prop split and train the classifier on everything
            XF = self.hlf.transform(X)
            y_outer = y
            
        else:
            # Inner high level features do need fitting
            # Stratified (hopefully) split of inner and outer training data
            X_inner, X_outer, y_inner, y_outer = sklearn.cross_validation.train_test_split(
                X, y, self._inner_prop=self._inner_prop, random_state=self.random_state)
            # Fit the inner, high level features
            self.hlf.fit(X_inner, y_inner)
            # Transform the training input for use with the outer classifier
            XF = self.hlf.transform(X_outer)
        
        # Resize if we have [num_aug, num_samples, num_features] shape
        if XF.ndim == 3:
            XF = np.reshape(XF, (XF.shape[0]*XF.shape[1], XF.shape[2]) )
        
        # Fit the outer classifier
        return self.clf.fit(XF, y_outer)
        
        
    def transform(self, X):
        """
        Transform X through inner and then outer classifier layers
        """
        # Might need to reshape here first...
        return self.clf.transform(self.hlf.transform(X))
        
        
    def fit_transform(self, X, y):
        """
        Fit to and then transform input data
        """
        self.fit(X,y)
        return self.transform(X,y)

