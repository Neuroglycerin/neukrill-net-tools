#!/usr/bin/env python
"""
Module for all image processing tools
"""

import skimage.io
import skimage.transform
import skimage.util
import numpy as np


def img_as_dtype(image, dt):
    """
    Convert an image to the target datatype.
    Serves as a wrapper for skimage.img_as_ubyte and
    skimage.img_as_float, but can be called to change
    datatype to dynamic target.
    """
    if dt == image.dtype:
        # Apparently they have the same type already
        return image
        
    elif dt == np.dtype(np.uint8):
        # (0, 255)
        return skimage.util.img_as_ubyte(image)
        
    elif dt == np.dtype(np.float64):
        # (0, 1)
        return skimage.util.img_as_float(image)
        
    elif dt == np.dtype(np.uint16):
        # (0, 65535)
        return skimage.util.img_as_uint(image)
        
    elif dt == np.dtype(np.int16):
        # (0, 32767)
        return skimage.util.img_as_int(image)
        
    elif dt == np.dtype(np.bool):
        # (False, True)
        return skimage.util.img_as_bool(image)
        
    else:
        raise ValueError('Unfathomable target datatype: %s' % dt)


def load_images(image_fpaths, processing, verbose=False):
    """
    Loads images provided in a list of filepaths
    and applies a processing function if supplied one.

    Processing function is expected to take a
    single argument, the image as a numpy array,
    and process it.
    """
    data_subset = []
    num_images = len(image_fpaths)

    for index in range(num_images):
        # read the image into a numpy array
        image = skimage.io.imread(image_fpaths[index])

        if verbose:
            print('image read: {0} of {1}'.format(index, num_images))

        if processing:
            # make sure that what we get out is a list
            # even if it's a single image
            resized_images = processing(image)
            if type(resized_images) is not list:
                resized_images = [resized_images]
            image_vectors = list(map(lambda image: image.ravel(),
                                            resized_images))
        else:
            image_vectors = [image.ravel()]

        data_subset += image_vectors

    return data_subset


def landscapise_image(image):
    """
    Makes sure the image is landscape.
    Takes the transpose of an image if it is portrait.
    In this case, landscape means dim1>=dim0
    """
    if len(image.shape) != 2:
        raise ValueError('Image does not have two dims')
    
    if image.shape[1] < image.shape[0]:
        return np.transpose(image)
        
    else:
        return image
    

def resize_image(image, size):
    """
    resize images to a pixel*pixel defined in a tuple
    input: image
           size e.g. the tuple (48,48)
    output: resized_image

    does this by padding to make the image square, then
    resizing
    """
    # First, use skimage to check what value white should be
    whiteVal = skimage.dtype_limits(image)[1]
    
    # find out how much we have to pad
    diff = abs(image.shape[1]-image.shape[0])
    
    # check if difference is divisible by two
    if diff%2 > 0:
        # not divisible, add extra pixel to one side
        extra = 1
        # then split to add to each side
        d = int((diff-1)/2)
    else:
        extra = 0
        d = int(diff/2)
    
    # which side to pad?
    if image.shape[1] > image.shape[0]:
        # more columns than rows, pad the rows
        padded_image = skimage.util.pad(image, ((d,d+extra),(0,0)),
                                        'constant', constant_values=whiteVal)
    else:
        # more rows than columns, pad the columns
        padded_image = skimage.util.pad(image, ((0,0),(d+extra,d)),
                                        'constant', constant_values=whiteVal)
    
    # Double-check we did make the image square
    if padded_image.shape[0] != padded_image.shape[1]:
        raise ValueError("Padded image is not square"
                            "Raise an issue about this."
                            "print rows: {0}, columns:{1}".format(
                        padded_image.shape[0],padded_image.shape[1]))

    # Now resize to specified size
    resized_image = skimage.transform.resize(padded_image, size)
    
    return resized_image


def flip_image(image, flip_x=False, flip_y=False):
    """
    Flips 2D images in either X or Y axis or both axes.
    Non-destructive: returns a copy of the input image
    input:  image  - input image
            flip_x - whether to flip image in X axis
            flip_y - whether to flip image in Y axis
    output: flipped_image - a new, transformed image
    """
    if len(image.shape) != 2:
        raise ValueError('Image must be 2-dimensional')

    flipped_image = image.copy()

    if flip_x:
        for row in range(flipped_image.shape[0]):
            flipped_image[row] = flipped_image[row][::-1]

    if flip_y:
        for column in range(flipped_image.shape[1]):
            flipped_image[:,column] = flipped_image[:,column][::-1]

    return flipped_image


def rotate_image(image, angle, resizable=True):
    """
    Rotates images by a given angle around its center. Points outside of the
    boundaries of the image are filled with value of the nearest point.
    Non-destructive: returns a copy of the input image
    input: image
           angle - rotation angle in degrees in counter-clockwise direction
    output: rotated_image - a new, transformed image.
            The datatype of the output will always match the input.
    
    NB. Rotations through 90 degree multiples are lossless. However, rotation
        through other angles requires interpolation. Skimage converts the
        image to float64 to have more accurate values, however this extra
        precision is lost if the input is uint8, since the output precision
        is reduced to match that of the input.
    """
    # Note down the original type
    original_dtype = image.dtype
    
    if abs(((angle+45) % 90)-45) < 1e-05:
        # Lossless cardinal rotation
        # Make sure we have a positive number of rotations
        angle = angle % 360
        # Rotate by 90 the correct number of times
        rotated_image = np.rot90(image, np.round(angle/90))
    else:
        # Use lossy rotation from skimage
        # We pad with white because that is the background
        rotated_image = skimage.transform.rotate(
                            image, angle,
                            resize=resizable,
                            mode='constant', cval=1.0)
        # NB: The output of the skimage rotation is always
        # a float, not ubyte
    
    # Preserve the datatype
    # Ensure output matches input
    rotated_image = img_as_dtype(rotated_image, original_dtype)
    
    return rotated_image


def crop_image(image, side_id, crop_proportion=0.2):
    """
    Crops a 2D image by a given amount from one of its sides.
    input:  image - input image
            side_id - which side to crop
                      0 right
                      1 top
                      2 left
                      3 right
            crop_proportion - how much to crop by (proportional to the
                              length of this side of the image)
    output: cropped_image - a new image, smaller on one side
    """
    if (side_id % 2)==0:
        # Left/right
        sidelength = image.shape[1]
    else:
        # Top/bottom
        sidelength = image.shape[0]
    
    croplen = np.floor(sidelength*crop_proportion)
    
    if side_id == 0:
        # RHS
        cropped_image = image[:, :-croplen]
    elif side_id == 1:
        # Top
        cropped_image = image[croplen:, :]
    elif side_id == 2:
        # LHS
        cropped_image = image[:, croplen:]
    elif side_id == 3:
        # Bottom
        cropped_image = image[:-croplen, :]
    else:
        raise ValueError('Side ID was not in [0,1,2,3]')
    return cropped_image


def padshift_image(image, centre_shift):
    """
    Pads an image so the centre shifts by the translation
    specified.
    input: image - input image
           centre_shift - amount to shift centre by
                          a two-element tuple of signed integers
    output: image - padded image with centre shifted by
                    centre_shift. This means the image is padded
                    with twice the abs of centre_shift on one/two
                    sides.
    """
    # First, use skimage to check what value white should be
    whiteVal = skimage.dtype_limits(image)[1]
    
    if centre_shift[0] < 0:
        # Negative shift of first dim
        # Move up by padding bottom
        padlen = -2 * centre_shift[0]
        pad = whiteVal * np.ones( (padlen, image.shape[1]) )
        image = np.concatenate( (image,pad), axis=0 )
        
    elif centre_shift[0] > 0:
        # Positive shift of first dim
        # Move down by padding top
        padlen = 2 * centre_shift[0]
        pad = whiteVal * np.ones( (padlen, image.shape[1]) )
        image = np.concatenate( (pad,image), axis=0 )

    if centre_shift[1] < 0:
        # Negative shift of second dim
        # Move left by padding afterward
        padlen = -2 * centre_shift[1]
        pad = whiteVal * np.ones( (image.shape[0], padlen) )
        image = np.concatenate( (image,pad), axis=1 )
        
    elif centre_shift[1] > 0:
        # Positive shift of second dim
        # Move right by padding before
        padlen = 2 * centre_shift[1]
        pad = whiteVal * np.ones( (image.shape[0], padlen) )
        image = np.concatenate( (pad,image), axis=1 )
    
    return image

def shape_fix(image, shape):
    """
    Makes all images the same size without resizing them.
    Crops large images down to their central SHAPE elements.
    Pads smaller images with white so the whole thing is sized SHAPE.
    The pixel value of "white" is determined from the datatype of
    the input.
    """
    # First, use skimage to check what value white should be
    whiteVal = skimage.dtype_limits(image)[1]
    # We will pad with 1.0 if input is float, 
    # or pad with 255 if input is ubyte
    
    # First do dim-0
    if image.shape[0] > shape[0]:
        # Too big; crop down
        start = np.floor( (image.shape[0] - shape[0])/2 )
        image = image[ start:(start+shape[0]) , : ]
    elif image.shape[0] < shape[0]:
        # Too small; pad up
        len0 = np.floor( (shape[0] - image.shape[0])/2 )
        pad0 = whiteVal * np.ones( (len0, image.shape[1]) )
        len1 = shape[0] - image.shape[0] - len0
        pad1 = whiteVal * np.ones( (len1, image.shape[1]) )
        image = np.concatenate( (pad0,image,pad1), axis=0 )
    
    # Now do dim-1
    if image.shape[1] > shape[1]:
        # Too big; crop down
        start = np.floor( (image.shape[1] - shape[1])/2 )
        image = image[ : , start:(start+shape[1]) ]
    else:
        # Too small; pad up
        len0 = np.floor( (shape[1] - image.shape[1])/2 )
        pad0 = whiteVal * np.ones( (image.shape[0], len0) )
        len1 = shape[1] - image.shape[1] - len0
        pad1 = whiteVal * np.ones( (image.shape[0], len1) )
        image = np.concatenate( (pad0,image,pad1), axis=1 )
    
    return image


def noisify_image(image, var=0.01, seed=None):
    """
    Adds Gaussian noise to image
    """
    # Note down the original type
    original_dtype = image.dtype
    
    # Apply Gaussian noise
    # NB: Output of skimage is always float64
    image = skimage.util.random_noise(image, seed=seed, var=var)
    
    # Preserve the image datatype
    # This will involve rounding pixels if the input was not float64
    image = img_as_dtype(image, original_dtype)
    
    return image
    

def mean_subtraction(image):
    """
    Sometimes useful to remove the per example mean:
    http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing
    """
    # for some reason, thought this would take more code
    return image - np.mean(image)

