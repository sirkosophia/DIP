# Code from: https://github.com/vpariza/open-hummingbird-eval


import random
from PIL import Image
# Custom function to apply resizing and cropping with probability
import torchvision.transforms.functional as F
from torchvision import transforms
import torch
import numpy as np
from PIL import ImageFilter

def random_resize_crop(image, target, size=(256, 256), scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
    ## convert target to tensor
    if not isinstance(target, torch.Tensor):
        target = transforms.ToTensor()(target)
    i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
    image = F.resized_crop(image, i, j, h, w, size, interpolation=Image.BILINEAR)
    target = F.resized_crop(target, i, j, h, w, size, interpolation=Image.NEAREST)
    return image, target

def resize(image, target, size=(256, 256)):
    ## convert target to tensor
    if not isinstance(target, torch.Tensor):
        target = transforms.ToTensor()(target)
    image = F.resize(image, size, interpolation=Image.BILINEAR)
    target = F.resize(target, size, interpolation=Image.NEAREST)
    return image, target


def apply_horizontal_flip(image, target):
    # Generate a random seed for the transformation
    if not isinstance(target, torch.Tensor):
        target = transforms.ToTensor()(target)
    seed = torch.randint(0, 2**32, size=(1,)).item()
    torch.manual_seed(seed)

    # Apply horizontal flip to the image
    image = F.hflip(image)

    # Use the same seed for the target to ensure consistent flip
    torch.manual_seed(seed)
    target = F.hflip(target)

    return image, target


class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.5, 2), ratio=(3. / 4., 4. / 3.), probability=1.0):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.probability = probability

    def __call__(self, img, target):
        if random.random() < self.probability:
            return random_resize_crop(img, target, self.size, self.scale, self.ratio)
        return img, target

class RandomHorizontalFlip(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img, target):
        if random.random() < self.probability:
            return apply_horizontal_flip(img, target)
        return img, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class Resize(object):
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img, target):
        return resize(img, target, self.size)


class CombTransforms(object):
    def __init__(self, img_transform=None, tgt_transform=None, img_tgt_transform=None):
        self.img_transform = img_transform
        self.tgt_transform = tgt_transform
        self.img_tgt_transform = img_tgt_transform

    def __call__(self, img, tgt):
        if self.img_transform:
            img = self.img_transform(img)
        if self.tgt_transform:
            tgt = self.tgt_transform(tgt)
        if self.img_tgt_transform:
            return self.img_tgt_transform(img, tgt)
        else:
            return img, tgt