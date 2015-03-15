"""
Dataset class that wraps the dataset class found in 
image_directory_dataset to support models with branched
input layers; allowing different versions of images as
input to those layers.

Based on dev work in the Interactive Pylearn2 notebook.
"""
__authors__ = "Scott Lowe"
__copyright__ = "Copyright 2015 - University of Edinburgh"
__credits__ = ["Scott Lowe"]
__license__ = "MIT"
__maintainer__ = "Scott Lowe"
__email__ = ""


import numpy as np
import sklearn.externals

import neukrill_net.image_directory_dataset
import neukrill_net.utils
import neukrill_net.encoding
import neukrill_net.image_processing
import skimage.util


class PreloadedPassthroughIterator(object):
    
    def __init__(self, dataset, batch_size, num_batches,
                 final_shape, rng, mode='even_shuffled_sequential',
                 train_or_predict="train", n_jobs=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.final_shape = final_shape
        self.train_or_predict = train_or_predict
        # initialise rng
        self.rng = rng
        # shuffle indices of size equal to number of examples
        # in dataset
        # indices of size equal to number of examples
        N = self.dataset.get_num_examples()
        self.indices = range(N)
        if mode == 'even_shuffled_sequential':
            self.rng.shuffle(self.indices)
        else:
            assert mode == 'even_sequential'
        # have to add this for checks during training
        # bit of a lie
        self.stochastic = False
        # this is also required
        self.num_examples = batch_size*num_batches
        
        # prepare first batch
        #self.batch_indices = [self.indices.pop(0) for i in range(self.batch_size)]
        # start the first batch computing
        #self.result = self.dataset.pool.map_async(self.dataset.fn,
        #                    [self.dataset.X[i] for i in self.batch_indices])

        # initialise flag variable for final iteration
        self.final_iteration = 0
        
    def __iter__(self):
        return self
    

    def next(self):
        # check if we reached the end yet
        if len(self.indices) >= self.batch_size:
            batch_indices = [self.indices.pop(0) 
                                for i in range(self.batch_size)]
        else:
            raise StopIteration
        
        # Work out which augmentation we are using for each image
        aug_ids = self.rng.randint(0,self.dataset.n_augment,(self.batch_size))
        
        # Use advanced indexing
        Xbatch = self.dataset.X[batch_indices, :, :, aug_ids]
        
        # Ensure connect unit type
        Xbatch = Xbatch.astype(np.float32)
        
        # Raise dimensionality
        if len(Xbatch.shape)<4:
            Xbatch = np.expand_dims(Xbatch, axis=3)
        
        if self.train_or_predict == "train":
            # get the batch for y as well
            ybatch = self.dataset.y[batch_indices,:].astype(np.float32)
        
        # Use advanced indexing for vbatch
        vbatch = self.dataset.cached[aug_ids, batch_indices, :]
        
        # start processing next batch
        if self.train_or_predict == "train":
            # get the batch for y as well
            return Xbatch,vbatch,ybatch
        elif self.train_or_predict == "test":
            return Xbatch,vbatch
        else:
            raise ValueError("Invalid option for train_or_predict:"
                    " {0}".format(self.train_or_predict))


class PreloadedPassthroughDataset(neukrill_net.image_directory_dataset.ListDataset):
    """
    Dataset that can supply arbitrary vectors as well as the Conv2D
    spaces required by the convolutional layers.
    """
    def __init__(self, settings_path="settings.json", 
                 run_settings_path="run_settings/alexnet_based.json",
                 training_set_mode="train", train_or_predict="train", 
                 verbose=False, force=False, cached=None):
        
        # ------ load settings
        
        # parse the settings file
        self.settings = neukrill_net.utils.Settings(settings_path)
        # get the run settings
        if train_or_predict == 'test':
            force=True
        self.run_settings = neukrill_net.utils.load_run_settings(run_settings_path,
                                                                self.settings,
                                                                force=force)
        
        # ------ load cache
        
        # Load the cached array
        self.cached = sklearn.externals.joblib.load(cached)
        
        # Check the number of augmentations matches
        assert self.run_settings["augmentation_factor"] == self.cached.shape[0]
        
        # get the right split from run settings
        li = neukrill_net.utils.train_test_split_bool(self.settings.image_fnames,
                                    training_set_mode,
                                    train_split=self.run_settings['train_split'],
                                    classes=self.settings.classes)
        
        print '-----------'
        print training_set_mode
        print len(li)
        print sum(li)
        print '-----------'
        
        # Use boolean indexing
        self.cached = self.cached[:,li,:] 
        
        # Make sure our array has the correct dtype
        self.cached = self.cached.astype(np.float32)
        
        # ------ done loading cache
        
        
        # ------ load the main augmented data
        
        # Get preprocessing 
        processing_settings = self.run_settings["preprocessing"]
        # get a processing function from this
        processing = neukrill_net.augment.augmentation_wrapper(**processing_settings)
        
        # super simple if statements for predict/train
        if train_or_predict == "train":
            # split the dataset based on training_set_mode option:
            self.settings.image_fnames[train_or_predict] = \
                    neukrill_net.utils.train_test_split(
                            self.settings.image_fnames, 
                            training_set_mode, 
                            train_split=self.run_settings["train_split"],
                            classes=self.settings.classes)
            
            # count the images
            self.N_images = sum([len(self.settings.image_fnames[train_or_predict][class_label])
                                for class_label in self.settings.classes])
            
            #load the data
            X_raw, labels = neukrill_net.utils.load_rawdata(
                            self.settings.image_fnames,
                            classes=self.settings.classes,
                            verbose=verbose)
            
            # initialise array
            X = np.zeros((self.N_images,self.run_settings["final_shape"][0],
                self.run_settings["final_shape"][1],self.run_settings["augmentation_factor"]))
            image_index = 0
            
            if self.run_settings.get("use_super_classes", False):
                # create dictionary to cache superclass vectors
                supclass_vecs = {}
                # get the general hierarchy
                general_hier = enc.get_hierarchy()
            
            for image_index, image in enumerate(X_raw):
                # apply processing function (get back multiple images)
                images = processing(image)
                # then broadcast each of these images into the empty X array
                for augment_index, augmented_image in enumerate(images):
                    X[image_index,:,:,augment_index] = augmented_image
            
            # if we're normalising
            if processing_settings.get("normalise", False):
                if verbose:
                    print("Applying normalisation: {0}".format(
                        processing_settings["normalise"]["global_or_pixel"]))
                # then call the normalise function
                X,self.run_settings = neukrill_net.image_processing.normalise(X,
                                            self.run_settings, verbose=verbose)
            
            
            self.X = X
            
            self.N = self.N_images
            self.n_classes = len(self.settings.classes)

            if self.run_settings.get("use_super_classes", False):
                supclass_vecs = {}
                general_hier = neukrill_net.encoding.get_hierarchy(self.settings)
                n_columns = sum([len(array) for array in general_hier])
                self.y = np.zeros((self.N,n_columns))
                class_dictionary = neukrill_net.encoding.make_class_dictionary(
                                            self.settings.classes,general_hier)
            else:
                self.y = np.zeros((self.N,self.n_classes))
                class_dictionary = {}
                for i,c in enumerate(self.settings.classes):
                    class_dictionary[c] = i
            for i,j in enumerate(map(lambda c: class_dictionary[c],labels)):
                self.y[i,j] = 1
            self.y = self.y.astype(np.float32)
            
            
        elif train_or_predict == "test":
            # test is just a big list of image paths
            # how many?
            self.N_images = len(self.settings.image_fnames[train_or_predict])

            # more boilerplate code, but it's going to be easier than making a
            # function that can deal with the above as well
            # initialise array
            X = np.zeros((self.N_images,self.run_settings["final_shape"][0],
                self.run_settings["final_shape"][1],self.run_settings["augmentation_factor"]))
            image_index = 0
            if verbose:
                print("Loading this many images:..........................")
                # get a list of 50 image_paths to watch out for
                stepsize = len(self.settings.image_fnames[train_or_predict]/50)
                progress_paths = [impath for i,impath in 
                        enumerate(self.settings.image_fnames[train_or_predict]) 
                        if i%stepsize == 0 ]
            # loop over all the images, in order
            for image_path in self.settings.image_fnames[train_or_predict]:
                if verbose:
                    if image_path in progress_paths: 
                        sys.stdout.write(".")
                        # if it's the last one we better stick a newline on
                        if image_path == progress_paths[-1]:
                            sys.stdout.write(".\n")
                # load the image as numpy array
                image = skimage.io.imread(image_path)
                # apply processing function (get back multiple images)
                images = processing(image)
                # then broadcast each of these images into the empty X array
                for image in images:
                    X[image_index,:,:,0] = image
                    image_index += 1
            # if we're normalising
            if processing_settings.get("normalise",0):
                if verbose:
                    print("Applying normalisation: {0}".format(
                        processing_settings["normalise"]["global_or_pixel"]))
                # then call the normalise function
                X,self.run_settings = neukrill_net.image_processing.normalise(X,
                                            self.run_settings, verbose=verbose)
            # store the names in this dataset object
            self.names = [os.path.basename(fpath) for fpath in 
                    self.settings.image_fnames[train_or_predict]]
            
            self.X = X
            self.N = self.N_images
            
        else:
            raise ValueError('Invalid option: should be either "train" for'
                             'training or "test" for prediction (I know '
                             ' that is annoying).')
         
        # ------ initialise
        
        assert self.N == self.X.shape[0] == self.cached.shape[1]
        
        #self.fn = transformer
        
        # initialise pool
        #self.pool = multiprocessing.Pool(n_jobs)
        
        self.n_augment = self.run_settings["augmentation_factor"]
        
        # count the classes
        self.n_classes = len(self.settings.classes)
        
        # set up the random state
        self.rng = np.random.RandomState(self.settings.random_seed)
        
        
    def iterator(self, mode=None, batch_size=None, num_batches=None, rng=None,
                        data_specs=None, return_tuple=False):
        """
        Returns iterator object with standard Pythonic interface; iterates
        over the dataset over batches, popping off batches from a shuffled 
        list of indices.
        Inputs:
            - mode: 'sequential' or 'shuffled_sequential'.
            - batch_size: required, size of the minibatches produced.
            - num_batches: supply if you want, the dataset will make as many
        as it can if you don't.
            - rng: not used, as above.
            - data_specs: not used, as above
            - return_tuple: not used, as above
        Outputs:
            - instance of FlyIterator, see above.
        """
        if not num_batches:
            # guess that we want to use all of them
            num_batches = int(len(self.X)/batch_size)
        iterator = PreloadedPassthroughIterator(dataset=self, batch_size=batch_size, 
                                num_batches=num_batches, 
                                final_shape=self.run_settings["final_shape"],
                                rng=self.rng, mode=mode)
        return iterator

