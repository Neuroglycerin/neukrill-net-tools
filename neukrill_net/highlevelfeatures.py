#!/usr/bin/env python
"""
Module for feature extractor object classes and classifier functions.
"""

# Rule for methods:
# When the input is images, it might be a list of np arrays, or a list of paths
# When the input is image, it is always an np array
# The base class handles conversion from paths to images in transform
# fit(images) will need to load and preprocess each element in images


from __future__ import division

import numpy as np
import cv2
import sklearn.cluster
import six
import skimage.io
import skimage.util
import mahotas.features
import neukrill_net.image_features
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
    
    _is_combiner = False
    
    def __init__(self, preprocessing_func=None, augment_func=None, **kwargs):
        """
        Initialise the preprocessing and augmentation functions.
        
        NOTE: should not be overwritten by children.
        """
        if not preprocessing_func is None:
            self.preprocess_image = preprocessing_func
            
        if not augment_func is None:
            self.augment_image = augment_func
        
    
    def __add__(self, other):
        """
        """
        if self._is_combiner and other._is_combiner:
            # Add all the other's children to me
            for child in other.childHLFs:
                self.add_HLF(child)
            return self
            
        elif self._is_combiner:
            # Add the HLF into me
            self.add_HLF(other)
            return self
            
        elif other._is_combiner:
            # Add me HLF to the combined HLF
            self.add_HLF(other)
            return other
            
        else:
            return MultiHighLevelFeature([self, other])
        
        
    def fit(self, images, y=[]):
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
        
        
    def extract_image(self, image):
        """
        Extracts a feature vector from the image
        Input : image - a 2D numpy array of an image
        Output: features - a 1D numpy array of features extracted from the image.
        """
        # Subclasses will need to overwrite this function
        raise NotImplementedError
        
        
    def preprocess_and_extract_image(self, image):
        """
        Preprocess image and than extract features
        Input : image - a 2D numpy array of an image
        Output: features - a 1D numpy array of features extracted from the image.
        
        NOTE: Subclasses should not modify this function!
        """
        return self.preprocess_image(self.extract_image(image))
        
        
    def transform(self, images):
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
        num_feature_elements = self.preprocess_and_extract_image(first_image).size
        
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
                # Extract features and put them into the array
                X[augment_index, image_index, :] = self.preprocess_and_extract_image(augmented_image).ravel()
        
        # Return the completed arary
        return X


class MultiHighLevelFeature(HighLevelFeatureBase):
    """
    Class for merging high level features together
    """
    
    _is_combiner = True
    
    def __init__(self, HLF_list, *args, **kwargs):
        """
        Initialise
        List of child high-level-features
        Input : HLF_list - List of high-level-feature objects
        """
        
        HighLevelFeatureBase.__init__(self, *args, **kwargs)
        
        self._childHLFs = HLF_list
        
        
    def add_HLF(self, HighLevelOther):
        """
        Add a new high level feature to the container
        """
        _childHLFs += [HighLevelOther]
        
        
    def fit(self, *args, **kwargs):
        """
        Fit each of the children features to a training set.
        
        Input : a list of images or image paths
        Output: None
        """
        # Fit each of the children in turn
        for child in self._childHLFs:
            child.fit(*args, **kwargs)
        
        
    def preprocess_image(self, image):
        """
        Preprocessing function does not make sense,
        since the features have different functions
        """
        # By default, the preprocessing function does nothing to the 
        # input image
        raise NotImplementedError
        
        
    def preprocess_and_extract_image(self, image):
        """
        Preprocess image and than extract features with each child
        Input : image - a 2D numpy array of an image
        Output: features - a 1D numpy array of features extracted from the image
                           with each HLF child in turn
        
        NOTE: Subclasses should not modify this function!
        """
        return np.concatenate( [child.preprocess_and_extract_image(image).ravel() for child in self._childHLFs] )



class BasicAttributes(HighLevelFeatureBase):
    """
    Get generic, basic, high level attributes from the image
    """
    def __init__(self, attributes_list, *args, **kwargs):
        
        HighLevelFeatureBase.__init__(self, *args, **kwargs)
        
        # Set the feature extractor function to provide the target list
        # of attributes
        self.extract_image = self.attributes_function_generator(attributes_list)
        
        
    @staticmethod
    def attributes_function_generator(attributes_list):
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
        
        HighLevelFeatureBase.__init__(self, **options)
        
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
    
    
    def fit(self, images, y=[]):
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
        
        
    def extract_image(self, image):
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


class Haralick(HighLevelFeatureBase):
    """
    Compute Haralick texture features
    """
    def __init__(self, preprocessing_func=skimage.util.img_as_ubyte, **options):
        """Initialisation"""
        HighLevelFeatureBase.__init__(self, **options)
        
    
    def extract_image(self, image):
        H  = mahotas.features.haralick(image)
        X0 = np.mean(H, 0)
        X1 = (np.amax(H, 0) - np.amin(H, 0))
        return np.concatenate((X0,X1))



class CoocurProps(HighLevelFeatureBase):
    """
    Compute texture features from Grey-Level Co-ocurance matrix properties
    """
    def __init__(self, preprocessing_func=skimage.util.img_as_ubyte, max_dist=18, num_angles=4, props=None, **options):
        """Initialisation"""
        HighLevelFeatureBase.__init__(self, **options)
        
        self.max_dist = max_dist
        self.num_angles = num_angles
        
        if props is None:
            props = ['contrast','dissimilarity','homogeneity','ASM','energy','correlation']
        self.props = props
    
    
    def extract_image(self, image):
        """
        Compute Grey-Level Co-ocurance matrix properties for a single image
        """
        P = np.zeros( (len(self.props), self.max_dist) )
        
        angles = np.arange(self.num_angles) * 2 * np.pi / self.num_angles
        GLCM = skimage.feature.greycomatrix(image, range(1,self.max_dist+1), angles,
                    levels=256, symmetric=False, normed=True)
        
        for prop_index, prop in enumerate(self.props):
            P[prop_index, :] = np.mean(skimage.feature.greycoprops(GLCM, prop=prop), 1)
        
        return P



class ContourMoments(HighLevelFeatureBase):
    """
    Compute moments of image (after segmentation)
    """
    def __init__(self, return_only_hu=False, **kwargs):
        """
        Initialise
        """
        # Call superclass
        HighLevelFeatureBase.__init__(self, **kwargs)
        
        self.return_only_hu = return_only_hu
        
        
    def extract_image(self, image):
        moments = neukrill_net.image_features.get_shape_moments(image)
        hu_moments = neukrill_net.image_features.get_shape_HuMoments(moments)
        if self.return_only_hu:
            return hu_moments
        else:
            return np.concatenate((np.array(hu_moments),np.array(moments.values())))
        

class KeypointEnsembleClassifier(HighLevelFeatureBase):
    """
    Classifies an image using the ensemble of descriptions of keypoints in the
    image.
    """
    
    num_classes = 0
    
    def __init__(self, detector, describer, classifier, return_num_kp=True, summary_method='mean', **kwargs):
        """
        Initialise the keypoint evidence tree
        """
        # Call superclass
        HighLevelFeatureBase.__init__(self, **kwargs)
        
        self.detector = detector
        self.describer = describer
        self.classifier = classifier
        self.return_num_kp = return_num_kp
        self.summary_method = summary_method
        
        self.scaler = sklearn.preprocessing.StandardScaler()
        
        
    def detect_and_describe(self, image):
        """
        Describe all the keypoints in an image
        Input : image - an image as a numpy array
        Output: descriptions - a numpy array of keypoint descriptions
                                sized (num_keypoints, description_len)
        """
        a,b,c = self.detector(image)
        return self.describer(a,b,**c)
        
        
    def describe_stack(self, images, y):
        """
        Describe all the keypoints in all the listed images
        Input : images - list of images or paths to images
        Output: descriptions - a numpy array of keypoint descriptions
                                sized (total_num_keypoints, description_len)
        """
        # Initialise
        descriptions = None
        y_full = []
        # Loop over list of images
        for image_index,image in enumerate(images):
            # Load the image if necessary
            image = loadimage(image)
            # Augment the image, giving a list of copies
            augmented_list = self.augment_image(image)
            # Loop over all the augmented copies
            for augment_index, augmented_image in enumerate(augmented_list):
                # Have to preprocess the augmented image
                augmented_image = self.preprocess_image(augmented_image)
                # Extract keypoint descriptions and put them into the array
                my_descriptions = self.detect_and_describe(augmented_image)
                if my_descriptions==[] or my_descriptions.size==0:
                    # do nothing
                    continue
                elif descriptions is None:
                    # Initialise with right shape
                    descriptions = my_descriptions
                else:
                    # Add to matrix
                    descriptions = np.concatenate((descriptions,my_descriptions))
                # Add to y
                y_full += [y[image_index]] * my_descriptions.shape[0]
                
        return descriptions, y_full
        
        
    def fit(self, images, y):
        """
        Fit the keypoint classifier to training data
        Input : images - list of images or image paths
                y - class labels
        Output: None
        """
        # Get keypoint descriptions for all the training data
        X, y = self.describe_stack(images, y)
        # Fit scaler against this distribution of keypoint descriptions
        self.scaler.fit(X)
        # Scale
        X = self.scaler.transform(X)
        # Fit the classifier
        self.classifier.fit(X, y)
        # Note the number of classes for later
        self.num_classes = len(np.unique(y))
        
        
    def extract_image(self, image):
        """
        Extract keypoint evidence from an image
        Input : image - image as a numpy array
        Output: vec - the feature vector
        """
        # Get descriptions
        descriptions = self.detect_and_describe(image)
        
        # Count how many keypoints we have
        num_kp = descriptions.shape[0]
        
        if num_kp==0:
            # Handle edge case where no keypoints are detected
            if self.summary_method=='mean':
                vec = np.ones((1,self.num_classes)) / num_kp
            else:
                vec = np.zeros((1,self.num_classes))
        else:
            # Scale descriptions
            descriptions = self.scaler.transform(descriptions)
            # Compute probabilites for each keypoint belonging to each class
            kp_probs = self.classifier.predict_proba(descriptions)
            
            # Average over keypoints
            if self.summary_method=='mean':
                # Take the mean of their probabilites
                vec = np.mean(kp_probs, 0)
                
            elif self.summary_method=='vote':
                # Let each keypoint vote, and take a probability distribution from
                # the votes
                vec = np.argmax(kp_probs, axis=1)
                vec, _ = np.histogram(vec, bins=np.arange(0,kp_probs.shape[1]+1)-0.5, density=True)
                
            else:
                raise ValueError("Unrecognised summary method: {}".format(self.summary_method))
                
            
        # Remove spare dimension
        vec = vec.ravel()
        
        if self.return_num_kp:
            # Add the number of keypoints to the start of the vector
            vec = np.concatenate((np.array([num_kp]),vec))
        
        return vec

