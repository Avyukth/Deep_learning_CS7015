import torch
import torch.nn as nn
from PIL import Image, ImageOps
import numpy as np
import random
import math
import shutil
from matplotlib import pyplot as plt

def save_model(model_state, is_best_acc, filename):
    """ Save model """
    # TODO: add it as checkpoint
    torch.save(model_state,filename)
    if is_best_acc:
        shutil.copyfile(filename, 'model_best.pth.tar')


def plotNNFilter(units,fname):
    filters = units.shape[3]
    fig=plt.figure(1, figsize=(80,80))
    n_columns = 8
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        ax=fig.add_subplot(n_rows, n_columns, i+1)
        # plt.title('Filter ' + str(i))
        data=units[i,:,:,:].reshape(3,3)
        # print(data)
        # img = Image.fromarray(data.astype(np.uint8), 'L')
        ax.imshow(data, interpolation="nearest", cmap="gray")
        fig.savefig(fname+'.png')

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


class RandomErasing(object):
    def __init__(self, EPSILON=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.EPSILON = EPSILON
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.EPSILON:
            return img
        img = np.array(img)
        area = img.shape[0] * img.shape[1]

        target_area = random.uniform(self.sl, self.sh) * area
        aspect_ratio = random.uniform(self.r1, 1 / self.r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img.shape[1] and h < img.shape[1]:
            x1 = random.randint(0, img.shape[0] - h)
            y1 = random.randint(0, img.shape[1] - w)
            img[x1:x1 + h, y1:y1 + w] = self.mean[0]
            img[x1:x1 + h, y1:y1 + w] = self.mean[1]

        img = Image.fromarray(img.astype("uint8"))
        return img