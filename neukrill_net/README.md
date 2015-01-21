There are currently in this repo:

* `utils.py`: various miscellaneous utilities, including data loading.
* `augment.py`: wrappers and functions for data augmentation, primarily
used to pass into the loading function.
* `image_processing.py`: used by `augment.py` for image processing. Intended
to process single images (as opposed to producing lists of images from single
images - which is what the functions in `augment.py` do).
* `constants.py`: contains some constants that can just be imported, such as 
the classes.

utils.py
========

Contains utilities for loading the data and loading the settings file. Is 
expected to be used for both training the model and prediction.

For example, the `Settings` class is instantiated in the main `train.py` 
scripts to provide a dictionary of the image filenames and directories
to hand to the data loading utility, `load_data`:

```python
    settings = utils.Settings('settings.json')

    # get all training file paths and class names
    image_fname_dict = settings.image_fnames
```

The `load_data` function itself takes this dictionary of image filenames
and loads the data, returning a training data matrix along with a vector
of labels if it finds training data or a test data matrix along with the
corresponding filenames if it finds test data.

Applying processing at load time
--------------------------------

For versatility, and to speed things up by applying processing in the data
loading loop, processing is applied by handing this function a data 
processing function called `processing`. _It is assumed that this function 
will take an array (an image) as an argument and return an array or
multiple arrays_.

This is for data augmentation and preprocessing at load time. It could
also be used to arbitrarily process the data at load time into visual
bag-of-words feature vectors or anything else, as long as it takes
an image as an array as an argument.

image_processing.py
===================

This contains functions that act on the images themselves. Practically, this
means that this is _where we import `skimage`_.

processing function
-------------------

This is also where the processing function is actually applied to the images,
and it deals with new images being created by the augmentation functions.

augment.py
==========

This is intended to contain any functions that will augment the data. 
Currently, it contains a wrapper for creating simple functions to resize and
rotate the data.

