"""
Unit tests for image processing functions
"""
import skimage.io
import skimage.transform
import numpy as np
from neukrill_net.tests.base import BaseTestCase
import neukrill_net.image_processing as image_processing


class TestLoadImages(BaseTestCase):
    """
    Unit tests for LoadImage function
    """
    def setUp(self):
        """
        Grab fpaths from base fpaths dict for test images
        """
        self.image_fpaths = self.image_fname_dict['test']

    #def test_load_images_without_processing(self):
    #    """
    #    Test load images returns list of flat images as expected
    #    with no processing
    #    """
    #    # Test we can load the images without a processing function
    #    images = image_processing.load_images(self.image_fpaths, None)
    #    self.assertEqual(len(images), 3)
        
    def test_load_images_with_min_processing(self):
        """
        Ensure a processing function works
        """
        processing = lambda image: image.min()
        images = image_processing.load_images(self.image_fpaths, processing)
        self.assertEqual(len(images), 3)
        self.assertEqual([[int(x)] for x in images], [[63], [5], [46]])


class TestLandscapise(BaseTestCase):
    """
    Unit tests for ensuring image is landscape
    """
    def setUp(self):
        """
        Make a dummy image
        """
        self.image = np.zeros((50,75), dtype=np.float64)
        self.v1 = np.array([[.1,.2,.3,.4]], dtype=np.float64)
        self.v0 = np.transpose(self.v1)
        self.image255 = np.array([[200,12,128],[42,64,196]], dtype=np.uint8)
    
    def test_landscapise(self):
        """
        Test landscapise
        """
        self.assertEqual(self.v1, image_processing.landscapise_image(self.v1))
        self.assertEqual(self.v1, image_processing.landscapise_image(self.v0))
        with self.assertRaises(ValueError):
            image_processing.landscapise_image(np.array([0,1]))

class TestResize(BaseTestCase):
    """
    Unit tests for image resizing as this function is going to expand
    """
    def setUp(self):
        """
        Read the first of the images using skimage
        """
        self.image  = skimage.io.imread(self.image_fname_dict['test'][0])
        self.image2 = np.ones((100,100), dtype=np.float64)
        self.image3 = 40 * np.ones((48,48), dtype=np.uint8)

    def test_resize(self):
        """
        Ensure resizing works
        """
        # Check we do nothing to the image if we resize to the same size
        #self.assertEqual(
        #    self.image,
        #    image_processing.resize_image(self.image, self.image.shape)
        #    )
        # Check sizes are okay
        self.assertEqual(
            (5,5),
            image_processing.resize_image(self.image, (5,5)).shape
            )
        self.assertEqual(
            (2000,2000),
            image_processing.resize_image(self.image, (2000,2000)).shape
            )
        # Check image can be shrunk correctly
        self.assertEqual(
            image_processing.resize_image(self.image2, (93,93)),
            np.ones((93,93), dtype=np.float64)
            )
        self.assertEqual(
            image_processing.resize_image(self.image3, (24,24)),
            40 * np.ones((24,24), dtype=np.uint8)
            )
        # Note: values are not preserved by upscaling.

class TestRescale(BaseTestCase):
    """
    Unit tests for image scaling
    """
    def setUp(self):
        """
        Read the first of the images using skimage
        """
        self.image  = skimage.io.imread(self.image_fname_dict['test'][0])
        self.image2 = np.ones((100,100), dtype=np.float64)
    
    def test_rescale(self):
        # Check we maintain exactly the same matrix if scale factor is 1.0
        self.assertEqual(
            self.image, image_processing.scale_image(self.image, 1.0)
            )
        self.assertEqual(
            self.image2, image_processing.scale_image(self.image2, 1.0)
            )
        # Check image doubles in size
        self.assertEqual(
            (200,200), image_processing.scale_image(self.image2, 2.0).shape
            )
        # Check image halves in size
        self.assertEqual(
            (50,50), image_processing.scale_image(self.image2, 0.5).shape
            )
        # Check image content matches when doubled
        self.assertEqual(
            np.ones((200,200), dtype=np.float64),
            image_processing.scale_image(self.image2, 2.0)
            )
        # Check image content matches when halved
        self.assertEqual(
            np.ones((50,50), dtype=np.float64),
            image_processing.scale_image(self.image2, 0.5)
            )

class TestRotate(BaseTestCase):
    """
    Unit tests for image rotation
    """
    def setUp(self):
        """
        Read the first of the images using skimage
        """
        self.image = skimage.io.imread(self.image_fname_dict['test'][0])
    
    def test_rotate_lossless(self):
        """
        Ensure rotation is lossless for cardinal rotations
        """
        # Rotation should be the same, mod 360
        self.assertEqual(
            image_processing.rotate_image(self.image, 90),
            image_processing.rotate_image(self.image, -270)
            )
        # Rotation should be the same, mod 360
        self.assertEqual(
            image_processing.rotate_image(self.image, 180),
            image_processing.rotate_image(self.image, -180)
            )
        # Rotation of 360 should not change image
        self.assertEqual(
            self.image,
            image_processing.rotate_image(self.image, 360)
            )
        # Rotation should stack 90+90=180
        self.assertEqual(
            image_processing.rotate_image(self.image, 180),
            image_processing.rotate_image(image_processing.rotate_image(self.image, 90), 90)
            )
        # Rotation of 90+270=360 should not change image
        self.assertEqual(
            self.image,
            image_processing.rotate_image(image_processing.rotate_image(self.image, 90), 270)
            )
        # Rotation of 180+180=360 should not change image
        self.assertEqual(
            self.image,
            image_processing.rotate_image(image_processing.rotate_image(self.image, 180), 180)
            )
        

class TestShear(BaseTestCase):
    """
    Unit tests for shearing
    """
    def setUp(self):
        """
        Read the first of the images using skimage
        """
        self.image  = skimage.io.imread(self.image_fname_dict['test'][0])
        self.image2 = np.ones((100,100), dtype=np.float64)
    
    def test_shear(self):
        # Check image remains the same with shear=0
        self.assertEqual(
            self.image,
            image_processing.shear_image(self.image, 0)
            )
        self.assertEqual(
            self.image2,
            image_processing.shear_image(self.image2, 0)
            )
        
    
class TestFlip(BaseTestCase):
    """
    Unit tests for image flipping
    """
    def setUp(self):
        """
        Read the first of the images using skimage
        """
        self.image = skimage.io.imread(self.image_fname_dict['test'][0])


    def check_images_are_equal(self, image, flipped_image,
                               flip_x=False, flip_y=False):
        """
        Checks whether the values in the two images are equal
        Can check for indices flipped around either axis
        """
        for x in range(image.shape[0]):
            rev_x = x if not flip_y else image.shape[0] - x - 1
            for y in range(image.shape[1]):
                rev_y = y if not flip_x else image.shape[1] - y - 1
                self.assertEqual(image[x, y], flipped_image[rev_x, rev_y])

    def test_flip(self):
        """
        Ensure flipping works
        """
        # Check when flipped in no axes
        flipped_image_x = image_processing.flip_image(self.image)

        self.check_images_are_equal(self.image, flipped_image_x)

        # Check when flipped in X-axis
        flipped_image_x = image_processing.flip_image(self.image, flip_x=True)

        self.check_images_are_equal(self.image, flipped_image_x, flip_x=True)
        
        # Check X flipping is reversible
        self.assertEqual(
            self.image,
            image_processing.flip_image(flipped_image_x, flip_x=True))

        # Check when flipped in Y-axis
        flipped_image_y = image_processing.flip_image(self.image, flip_y=True)
        
        self.check_images_are_equal(self.image, flipped_image_y, flip_y=True)
        
        # Check Y flipping is reversible
        self.assertEqual(
            self.image,
            image_processing.flip_image(flipped_image_y, flip_y=True))
        
        # Check when flipped in X- & Y-axes
        flipped_image_xy = \
                image_processing.flip_image(self.image,
                                            flip_x=True,
                                            flip_y=True)

        self.check_images_are_equal(self.image, flipped_image_xy, flip_x=True, flip_y=True)
        
        # Check X & Y flipping is reversible
        self.assertEqual(
            self.image,
            image_processing.flip_image(flipped_image_xy, flip_x=True, flip_y=True))

        
class TestCrop(BaseTestCase):
    """
    Unit tests for image flipping
    """
    def setUp(self):
        """
        Make a dummy image
        """
        self.image = np.ones((100,100))
        
    def test_crop(self):
        """
        Ensure cropping works.
        """
        # Crop 20% from right hand side
        croppedImage = image_processing.crop_image(self.image, 0, crop_proportion=0.2)
        self.assertEqual(croppedImage.shape, (100,80))
        # Crop 20% from top
        croppedImage = image_processing.crop_image(self.image, 1, crop_proportion=0.2)
        self.assertEqual(croppedImage.shape, (80,100))
        # Crop 20% from left hand side
        croppedImage = image_processing.crop_image(self.image, 2, crop_proportion=0.2)
        self.assertEqual(croppedImage.shape, (100,80))
        # Crop 20% from bottom
        croppedImage = image_processing.crop_image(self.image, 3, crop_proportion=0.2)
        self.assertEqual(croppedImage.shape, (80,100))
        
        # Crop 5.4% from right hand side
        croppedImage = image_processing.crop_image(self.image, 0, crop_proportion=0.054)
        self.assertEqual(croppedImage.shape, (100,95))
        # Crop 5.4% from top hand side
        croppedImage = image_processing.crop_image(self.image, 1, crop_proportion=0.054)
        self.assertEqual(croppedImage.shape, (95,100))


class TestPadShift(BaseTestCase):
    """
    Unit tests for image padding to shift centre around
    """
    def setUp(self):
        """
        Make a dummy image
        """
        self.image = np.zeros((50,75), dtype=np.float64)
        self.v1 = np.array([[.1,.2,.3,.4]], dtype=np.float64)
        self.v0 = np.transpose(self.v1)
        self.image255 = np.array([[200,12,128],[42,64,196]], dtype=np.uint8)
        
    def test_pad(self):
        """
        Ensure padding happens
        """
        # Ensure we return the same without padding
        paddedImage = image_processing.padshift_image(self.image255, (0,0))
        self.assertEqual(paddedImage, self.image255)
        # Test for shift upwards
        paddedImage = image_processing.padshift_image(self.v0, (-1,0))
        self.assertEqual(paddedImage, np.array([[.1],[.2],[.3],[.4],[1],[1]], dtype=np.float64))
        # Test for shift downwards
        paddedImage = image_processing.padshift_image(self.v0, (1,0))
        self.assertEqual(paddedImage, np.array([[1],[1],[.1],[.2],[.3],[.4]], dtype=np.float64))
        # Test for shift leftwards
        paddedImage = image_processing.padshift_image(self.v1, (0,-1))
        self.assertEqual(paddedImage, np.array([[.1,.2,.3,.4,1,1]], dtype=np.float64))
        # Test for shift rightwards
        paddedImage = image_processing.padshift_image(self.v1, (0,1))
        self.assertEqual(paddedImage, np.array([[1,1,.1,.2,.3,.4]], dtype=np.float64))
        # Test for uint8
        paddedImage = image_processing.padshift_image(self.image255, (-1,1))
        self.assertEqual(
            paddedImage,
            np.array([
                    [255,255,200, 12,128],
                    [255,255, 42, 64,196],
                    [255,255,255,255,255],
                    [255,255,255,255,255]
                ], dtype=np.uint8)
            )


class TestShapeFix(BaseTestCase):
    """
    Unit tests for shape_fix
    """
    def setUp(self):
        """
        Make a dummy image
        """
        self.image = np.zeros((50,75), dtype=np.float64)
        self.v1 = np.array([[.1,.2,.3,.4]], dtype=np.float64)
        self.v0 = np.transpose(self.v1)
        self.image255 = np.array([[200,12,128],[42,64,196]], dtype=np.uint8)
    
    def test_shapefix(self):
        """
        Ensure shape fixing works.
        """
        # Reshape to same shape as original
        # Should be left unchanged
        reshapedImage = image_processing.shape_fix(self.image, self.image.shape)
        self.assertEqual(reshapedImage.shape, self.image.shape)
        self.assertEqual(reshapedImage, self.image)
        # Test reducing dim-0 only
        reshapedImage = image_processing.shape_fix(np.ones((75,50), dtype=np.float64), (50,50))
        self.assertEqual(reshapedImage, np.ones((50,50), dtype=np.float64))
        # Test reducing dim-1 only
        reshapedImage = image_processing.shape_fix(np.ones((40,80), dtype=np.float64), (40,20))
        self.assertEqual(reshapedImage, np.ones((40,20), dtype=np.float64))
        # Test reducing dim-0 and dim-1
        reshapedImage = image_processing.shape_fix(np.ones((20,40), dtype=np.float64), (19,19))
        self.assertEqual(reshapedImage, np.ones((19,19), dtype=np.float64))
        # Test expanding dim-0 only
        reshapedImage = image_processing.shape_fix(np.ones((30,50), dtype=np.float64), (50,50))
        self.assertEqual(reshapedImage, np.ones((50,50), dtype=np.float64))
        # Test expanding dim-1 only
        reshapedImage = image_processing.shape_fix(np.ones((40,20), dtype=np.float64), (30,30))
        self.assertEqual(reshapedImage, np.ones((30,30), dtype=np.float64))
        # Test expanding dim-0 and dim-1
        reshapedImage = image_processing.shape_fix(np.ones((10,14), dtype=np.float64), (16,16))
        self.assertEqual(reshapedImage, np.ones((16,16), dtype=np.float64))
        # Test crop is centred for dim-0
        reshapedVect = image_processing.shape_fix(self.v0, (2,1))
        self.assertEqual(reshapedVect, np.array([[.2],[.3]], dtype=np.float64))
        # Test crop is centred for dim-1
        reshapedVect = image_processing.shape_fix(self.v1, (1,2))
        self.assertEqual(reshapedVect, np.array([[.2,.3]], dtype=np.float64))
        # Test pad is centred for dim-0
        reshapedVect = image_processing.shape_fix(self.v0, (6,1))
        self.assertEqual(reshapedVect, np.array([[1],[.1],[.2],[.3],[.4],[1]], dtype=np.float64))
        # Test pad is centred for dim-1
        reshapedVect = image_processing.shape_fix(self.v1, (1,6))
        self.assertEqual(reshapedVect, np.array([[1,.1,.2,.3,.4,1]], dtype=np.float64))
        # Test pad and crop works correctly
        reshapedImage = image_processing.shape_fix(self.v1, (3,2))
        self.assertEqual(reshapedImage, np.array([[1,1],[.2,.3],[1,1]], dtype=np.float64))
        # Test crop and padworks correctly
        reshapedImage = image_processing.shape_fix(self.v0, (2,3))
        self.assertEqual(reshapedImage, np.array([[1,.2,1],[1,.3,1]], dtype=np.float64))
        # Make sure output is correct for uint8 image too
        reshapedImage = image_processing.shape_fix(self.image255, (4,3))
        self.assertEqual(reshapedImage, np.array([[255,255,255],[200,12,128],[42,64,196],[255,255,255]], dtype=np.uint8))
        # again
        reshapedImage = image_processing.shape_fix(self.image255, (2,1))
        self.assertEqual(reshapedImage, np.array([[12],[64]], dtype=np.uint8))
        # again
        reshapedImage = image_processing.shape_fix(self.image255, (2,5))
        self.assertEqual(reshapedImage, np.array([[255,200,12,128,255],[255,42,64,196,255]], dtype=np.uint8))
    
    def test_padded_shapefix(self):
        """
        Ensure we get the appropriate output when combining with padding to move image within window.
        """
        paddedImage = image_processing.padshift_image(self.image255, (-1,1))
        reshapedImage = image_processing.shape_fix(paddedImage, (2,3))
        self.assertEqual(
            reshapedImage,
            np.array([
                    [255, 42, 64],
                    [255,255,255],
                ], dtype=np.uint8)
            )
        
