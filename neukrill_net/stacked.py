#!/usr/bin/env python
"""
Module hierarchically stacked classifiers
"""

from __future__ import division

import numpy as np
import sklearn.cross_validation
import copy


def get_leaves(hdic):
    """
    Give it a dictionary and it returns a list of all leaf nodes
    Input : hdic - a hierarchical dictionary
    Output: leaves - a list of leaf nodes
    """
    leaves = []
    for key, value in hdic.iteritems():
        if isinstance(value, dict):
            # Descend through the dictionary stack
            leaves += get_leaves(value)
        else:
            leaves += [value]
    return leaves


def propagate_labels_to_leaves(hdic, classes, mode='key->value'):
    """
    Inputs: hdic - a hierarchical dictionary
            classes - a list of class names
            mode - replacment mode
                   'key->key' : replace the key when the key is a class
                   'key->value' : replace the value when its key is a class
                   'value->value' : replace the value when the value is a class
    Output: hdic - a hierarchical dictionary where nodes in classes have been
                   replaced as per mode
    """
    hdic2 = {}
    for key, value in hdic.iteritems():
        if mode=='key->key' or mode=='key->value':
            # Check if the key is in the search terms
            found_index = None
            for class_index,class_name in enumerate(classes):
                if key==class_name:
                    found_index = class_index
                    break
            
            if mode=='key->key':
                if found_index is not None:
                    new_key = found_index
                else:
                    new_key = key
                    
                if isinstance(value, dict):
                    # Descend through the dictionary stack
                    hdic2[new_key] = propagate_labels_to_leaves(value, classes, mode=mode)
                else:
                    hdic2[new_key] = value
            
            if mode=='key->value':
                if found_index is not None:
                    hdic2[key] = found_index
                    
                elif isinstance(value, dict):
                    # Descend through the dictionary stack
                    hdic2[key] = propagate_labels_to_leaves(value, classes, mode=mode)
                
                else:
                    hdic2[key] = value
                    
            
        elif mode=='value->value':
            # Check if the value is in the search terms
            found_index = None
            for class_index,class_name in enumerate(classes):
                if value==class_name:
                    found_index = class_index
                    break
            
            if found_index is not None:
                hdic2[key] = found_index
                
            elif isinstance(value, dict):
                # Descend through the dictionary stack
                hdic2[key] = propagate_labels_to_leaves(value, classes, mode=mode)
                
            else:
                hdic2[key] = value
    
    return hdic2
    

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
        self.random_state = random_state
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
                X, y, test_size=self._inner_prop, random_state=self.random_state)
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
        
    
    def predict_proba(self, X):
        """
        Get probabilites from classifier
        """
        return self.clf.predict_proba(self.hlf.transform(X))


class HierarchyClassifier():
    """
    Hierarchical classifier
    """
    
    def __init__(self, class_hierarchy, clf):
        """
        Initialise.
        Inputs : class_hierarchy - a dictionary
        """
        self.class_hierarchy = class_hierarchy
        self.base_clf = clf
        
        
    def fit(self, X, y):
        """
        Fit model to training data X with labels y
        """
        # Stores the top-level tuple
        self.clf_hierarchy = self.subfit(X, y, self.class_hierarchy, self.base_clf)
        
        
    def subfit(self, X, y, hierarchy, base_clf):
        """
        Fit model for a single branch of a hierarchical classifier
        """
        # Ensure y is a numpy array
        y = np.array(y)
        # Initialise a new y, which we will relabel with each branch as a "class"
        my_y = -1 * np.ones(y.shape)
        # Initialise hierarchy
        clf_hierarchy = {}
        branch_counter = 0
        
        branch_map = {}
        
        # Loop over keys in dictionary
        for key, value in hierarchy.iteritems():
            if isinstance(value, dict):
                # Descend through the dictionary stack
                subhierarchy = value
                
                # Check which classes are at the leaves of this branch
                classes_in_sub = get_leaves(subhierarchy)
                
                # Reduce X and y down to just the correct classes
                # This is probably wrong...
                indices = np.concatenate([np.flatnonzero(y == class_name) for class_name in classes_in_sub])
                
                # Note for ourselves the label associated with the branch
                my_y[indices] = branch_counter
                
                # Reduce X and y down for child, so it only has relevant data
		if isinstance(X,list):
                     subX = [X[index] for index in indices]
		else:
                     subX = X[indices,:]
                suby = y[indices]
                
                # Request child branch do its own hierarchical training
                clf_hierarchy[branch_counter] = self.subfit(subX, suby, subhierarchy, base_clf)
                
            else:
                # This is a leaf node, so we don't need to go any deeper
                
                # Note for ourselves the label associated with the leaf
                my_y[y == value] = branch_counter
                
                # Record an empty hierarchy from this leaf
                clf_hierarchy[branch_counter] = (None, None, value)
                
            # Record this label->branch_name mapping
            branch_map[branch_counter] = key
            # Increment the branch counter
            branch_counter += 1
        
        # Ensure there are no unlabelled elements
        if any(y == -1):
            raise ValueError('Some samples were not relabelled')
        
        # Train our own classifier
        my_clf = copy.deepcopy(base_clf)
        my_clf.fit(X, my_y)
        
        return (my_clf, branch_map, clf_hierarchy)
        
        
    def transform(self, X):
        """
        Push X through the hierarchy
        """
        raise NotImplementedError
        
        
    def fit_transform(self, X, y):
        """
        Fit to and then transform input data
        """
        self.fit(X,y)
        return self.transform(X,y)
        
        
    def predict_proba(self, X):
        """
        Get probability of each X belonging to each class
        """
        p_dict = self.sub_predict_proba(X, *self.clf_hierarchy)
        # Turn dictionary into numpy array with correct ordering
        p = -1 * np.ones( (X.shape[0], len(p_dict)) )
        for class_label, class_proba in p_dict.iteritems():
            p[:,class_label] = class_proba
        return p
        
        
    def sub_predict_proba(self, X, my_clf, branch_map, clf_hierarchy):
        """
        Get the probability of membership of each branch at one level of the
        hierarchical model.
        Propagates probabilities throughout the hierarchy.
        Inputs: X - test dataset
                my_clf - classifier for this mother branch
                branch_map - map of classifier outputs to branch names
                clf_hierarchy - stacked dictionary of (clf,brmap,hier) tuples
        Output: p - probabilities of each sample belonging to possible classes
                    given it is a member of this mother branch
                    p is a dictionary mapping from output class_label to proba
        """
        # Initialise probabilities of membership of each class
        p = {}
        
        # Find the probability of each X being a member of each branch
        my_p = my_clf.predict_proba(X)
        
        # Loop over keys in dictionary
        for branch_label, branch_name in branch_map.iteritems():
            
            branch_value = clf_hierarchy[branch_label][2]
            
            if isinstance(branch_value, dict):
                # This is a branch.
                # Need to descend down the branch
                
                # Calling this function for the branch returns probability
                # X is in each overall class, given it was in the daughter branch
                sub_p = self.sub_predict_proba(X, *clf_hierarchy[branch_label])
                
                # Propagate the probability we are in this daughter branch onto
                # each of the overall class probabilities
                for key, value in sub_p.iteritems():
                    p[key] = my_p[:,branch_label] * value
                
            else:
                # This is a leaf node.
                # So we return its probability as-is.
                # Probability of leaf 'branch_value' given we are on mother-branch
                # is equal to classifier's probability output
                p[branch_value] = my_p[:,branch_label]
        
        return p
