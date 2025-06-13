# Code from: https://github.com/vpariza/open-hummingbird-eval

import torchvision.transforms as trn
import torch
import torch.nn.functional as F
from hummingbird.src.image_transformations import Compose, RandomResizedCrop, RandomHorizontalFlip, Resize

IMAGNET_MEAN = [0.485, 0.456, 0.406]
IMAGNET_STD = [0.229, 0.224, 0.255]

OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_STDN = [0.26862954, 0.26130258, 0.27577711]



def get_hbird_train_transforms_for_imgs(input_size = 224,
        min_scale_factor = 0.5,
        max_scale_factor = 2.0,
        brightness_jitter_range = 0.1,
        contrast_jitter_range = 0.1,
        saturation_jitter_range = 0.1,
        hue_jitter_range = 0.1,
        brightness_jitter_probability = 0.5,
        contrast_jitter_probability = 0.5,
        saturation_jitter_probability = 0.5,
        hue_jitter_probability = 0.5,
        img_mean = IMAGNET_MEAN,
        img_std = IMAGNET_STD):


    # 1. Image transformations for training
    image_train_global_transforms = [trn.RandomResizedCrop(size=(input_size, input_size), scale=(min_scale_factor, max_scale_factor))]
    image_train_local_transforms = [
        trn.RandomApply([trn.ColorJitter(brightness=brightness_jitter_range)], p=brightness_jitter_probability),
        trn.RandomApply([trn.ColorJitter(contrast=contrast_jitter_range)], p=contrast_jitter_probability),
        trn.RandomApply([trn.ColorJitter(saturation=saturation_jitter_range)], p=saturation_jitter_probability),
        trn.RandomApply([trn.ColorJitter(hue=hue_jitter_range)], p=hue_jitter_probability),
        trn.ToTensor(),
        trn.Normalize(mean=img_mean, std=img_std)
    ]
    image_train_transform = trn.Compose([*image_train_global_transforms, *image_train_local_transforms])

    # 3. Return the transformations in dictionaries for training and validation
    train_transforms = {"img": image_train_transform, "target": None, "shared": None}
    return train_transforms

def get_hbird_transforms(input_size = 224,
        min_scale_factor = 0.5,
        max_scale_factor = 2.0,
        brightness_jitter_range = 0.1,
        contrast_jitter_range = 0.1,
        saturation_jitter_range = 0.1,
        hue_jitter_range = 0.1,
        brightness_jitter_probability = 0.5,
        contrast_jitter_probability = 0.5,
        saturation_jitter_probability = 0.5,
        hue_jitter_probability = 0.5,
        img_mean = IMAGNET_MEAN,
        img_std = IMAGNET_STD,
        n_views = 1):
    # 1. Return the transformations in dictionaries for training, validation, and testing
    train_transforms = get_hbird_train_transforms(input_size, 
                                                    min_scale_factor, 
                                                    max_scale_factor, 
                                                    brightness_jitter_range, 
                                                    contrast_jitter_range, 
                                                    saturation_jitter_range, 
                                                    hue_jitter_range, 
                                                    brightness_jitter_probability, 
                                                    contrast_jitter_probability, 
                                                    saturation_jitter_probability, 
                                                    hue_jitter_probability, 
                                                    img_mean, 
                                                    img_std,
                                                    n_views)
    val_transforms = get_hbird_val_transforms(input_size, img_mean, img_std)
    test_transforms = get_hbird_val_transforms(input_size, img_mean, img_std)
    return train_transforms, val_transforms, test_transforms
def get_hbird_train_transforms_vis(input_size = 224,
        min_scale_factor = 0.5,
        max_scale_factor = 2.0,
        brightness_jitter_range = 0.1,
        contrast_jitter_range = 0.1,
        saturation_jitter_range = 0.1,
        hue_jitter_range = 0.1,
        brightness_jitter_probability = 0.5,
        contrast_jitter_probability = 0.5,
        saturation_jitter_probability = 0.5,
        hue_jitter_probability = 0.5,
        img_mean = IMAGNET_MEAN,
        img_std = IMAGNET_STD,
        n_views = 1):

    if n_views != 1:
        raise ValueError("Only 1 view is supported now")

    # 1. Image transformations for training
    image_train_transform = trn.Compose([
        trn.RandomApply([trn.ColorJitter(brightness=brightness_jitter_range)], p=brightness_jitter_probability),
        trn.RandomApply([trn.ColorJitter(contrast=contrast_jitter_range)], p=contrast_jitter_probability),
        trn.RandomApply([trn.ColorJitter(saturation=saturation_jitter_range)], p=saturation_jitter_probability),
        trn.RandomApply([trn.ColorJitter(hue=hue_jitter_range)], p=hue_jitter_probability),
        trn.ToTensor(),
        # trn.Normalize(mean=img_mean, std=img_std)
    ])

    # 2. Shared transformations for training
    shared_train_transform = Compose([
        RandomResizedCrop(size=(input_size, input_size), scale=(min_scale_factor, max_scale_factor)), #check this 
    ])

    # 3. Return the transformations in dictionaries for training and validation
    train_transforms = {"img": image_train_transform, "target": None, "shared": shared_train_transform}
    return train_transforms

def get_hbird_train_transforms(input_size = 224,
        min_scale_factor = 0.5,
        max_scale_factor = 2.0,
        brightness_jitter_range = 0.1,
        contrast_jitter_range = 0.1,
        saturation_jitter_range = 0.1,
        hue_jitter_range = 0.1,
        brightness_jitter_probability = 0.5,
        contrast_jitter_probability = 0.5,
        saturation_jitter_probability = 0.5,
        hue_jitter_probability = 0.5,
        img_mean = IMAGNET_MEAN,
        img_std = IMAGNET_STD,
        n_views = 1,
        is_radio = False):

    if n_views != 1:
        raise ValueError("Only 1 view is supported now")

    # 1. Image transformations for training
    if is_radio:
        image_train_transform = trn.Compose([
            trn.RandomApply([trn.ColorJitter(brightness=brightness_jitter_range)], p=brightness_jitter_probability),
            trn.RandomApply([trn.ColorJitter(contrast=contrast_jitter_range)], p=contrast_jitter_probability),
            trn.RandomApply([trn.ColorJitter(saturation=saturation_jitter_range)], p=saturation_jitter_probability),
            trn.RandomApply([trn.ColorJitter(hue=hue_jitter_range)], p=hue_jitter_probability),
            trn.ToTensor(),
            # trn.Normalize(mean=img_mean, std=img_std)
        ])
    else:
        image_train_transform = trn.Compose([
            trn.RandomApply([trn.ColorJitter(brightness=brightness_jitter_range)], p=brightness_jitter_probability),
            trn.RandomApply([trn.ColorJitter(contrast=contrast_jitter_range)], p=contrast_jitter_probability),
            trn.RandomApply([trn.ColorJitter(saturation=saturation_jitter_range)], p=saturation_jitter_probability),
            trn.RandomApply([trn.ColorJitter(hue=hue_jitter_range)], p=hue_jitter_probability),
            trn.ToTensor(),
            trn.Normalize(mean=img_mean, std=img_std)
        ])

    # 2. Shared transformations for training
    shared_train_transform = Compose([
        RandomResizedCrop(size=(input_size, input_size), scale=(min_scale_factor, max_scale_factor)), #check this 
        # RandomHorizontalFlip(probability=0.1),
    ])

    # 3. Return the transformations in dictionaries for training and validation
    train_transforms = {"img": image_train_transform, "target": None, "shared": shared_train_transform}
    return train_transforms


def get_hbird_val_transforms(input_size = 224,
                    img_mean = IMAGNET_MEAN,
                    img_std = IMAGNET_STD):
    return get_default_val_transforms(input_size, img_mean, img_std)

def get_hbird_val_transforms_vis(input_size = 224,
                    img_mean = IMAGNET_MEAN,
                    img_std = IMAGNET_STD):
    return get_default_val_transforms(input_size, None, None)

def get_default_train_transforms(input_size = 224,
                    img_mean = IMAGNET_MEAN,
                    img_std = IMAGNET_STD,
                    min_scale_factor = 0.5,
                    max_scale_factor = 2.0):
    # 1. Image transformations for training
    image_train_transform = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(mean=img_mean, std=img_std)
    ])

    # 2. Shared transformations for training
    shared_train_transform = Compose([
        RandomResizedCrop(size=(input_size, input_size), scale=(min_scale_factor, max_scale_factor)),
    ])
    # 3. Return the transformations in dictionary for training
    return {"img": image_train_transform, "target": None, "shared": shared_train_transform}

def get_default_val_transforms(input_size = 224,
                    img_mean = IMAGNET_MEAN,
                    img_std = IMAGNET_STD):
    # 1. Image transformations for validation
    if img_mean is None or img_std is None:
        image_val_transform = trn.Compose([trn.Resize((input_size, input_size)), trn.ToTensor()])
    else:
        image_val_transform = trn.Compose([trn.Resize((input_size, input_size)), trn.ToTensor(), trn.Normalize(mean=img_mean, std=img_std)])

    # 2. Shared transformations for validation
    shared_val_transform = Compose([
        Resize(size=(input_size, input_size)),
    ])

    # 3. Return the transformations in a dictionary for validation
    val_transforms = {"img": image_val_transform, "target": None , "shared": shared_val_transform}
    return val_transforms



def get_default_transforms(input_size = 224,
                    img_mean = IMAGNET_MEAN,
                    img_std = IMAGNET_STD):
    # 1. Return the transformations in dictionaries for training, validation, and testing
    train_transforms = get_default_train_transforms(input_size, img_mean, img_std)
    val_transforms = get_default_val_transforms(input_size, img_mean, img_std)
    test_transforms = get_default_val_transforms(input_size, img_mean, img_std)
    return train_transforms, val_transforms, test_transforms