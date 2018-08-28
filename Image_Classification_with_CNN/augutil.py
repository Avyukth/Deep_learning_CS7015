import numpy as np
import pandas as pd
from skimage import transform
from PIL import Image
import random
import skimage.io
from PIL import Image, ImageOps




class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if np.random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class RandomVerticalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if np.random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

class RandomRotation(object):
    """Rotate PIL.Image randomly (90/180/270 degrees)with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be rotated.
        Returns:
            PIL.Image: Randomly rotated image.
        """
        if np.random.random() < 0.5:
            deg = np.random.randint(1,3)*90.
            return img.rotate(deg)
        return img

class RandomTranslation(object):
    """Translates PIL.Image randomly (0-10 pixels) with a probability of 0.5."""

    def __init__(self,max_vshift=10, max_hshift=10):
        self.max_vshift = max_vshift
        self.max_hshift = max_hshift

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be translated.
        Returns:
            PIL.Image: Randomly translated image.
        """
        if np.random.random() < 0.5:
            hshift = np.random.randint(-self.max_hshift,self.max_hshift)
            vshift = np.random.randint(-self.max_vshift,self.max_vshift)
            return img.transform(img.size, Image.AFFINE, (1, 0, hshift, 0, 1, vshift))
        return img

    
hFlip=RandomHorizontalFlip()
vFlip=RandomVerticalFlip()
rRotatae=RandomRotation()
rTrans=RandomTranslation()    


def Hflip(im_image):
    out=hFlip(im_image)
    out_array=np.array(out).reshape(1,784)
    return out_array
    
    
def Vflip(im_image):
    out=vFlip(im_image)
    out_array=np.array(out).reshape(1,784)
    return out_array

def Rrot(im_image):
    out=rRotatae(im_image)
    out_array=np.array(out).reshape(1,784)
    return out_array

def Rtra(im_image):
    out=rTrans(im_image)
    out_array=np.array(out).reshape(1,784)
    return out_array

