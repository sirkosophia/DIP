"""
References:
    Croc: https://github.com/naver/croco
    
"""

import os
import ast
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# --- Helper Functions --- #

def load_img(file: str) -> Image.Image:
    """Load an image and ensure it is in RGB mode."""
    image = Image.open(file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def load_label(file: str) -> torch.Tensor:
    """Load a label tensor from a file."""
    return torch.load(file)

def convert_mask_path(original_path: str, mask_dir: str, data_dir_base: str) -> str:
    """
    Convert the given image path to a mask file path.
    
    Args:
        original_path (str): Original image file path.
        mask_dir (str): Folder name to insert into the new path.
        
    Returns:
        str: Converted mask file path.
    """
    new_path = original_path.replace(
    data_dir_base, mask_dir).replace('.JPEG', '.pt')
    return new_path

def load_pairs_from_list_file(fname: str, root: str = '', split: str = "train") -> list:
    """
    Parse a list file to obtain image pairs.
    
    Each line in the file should have two paths separated by ':'.
    For lines where the second path is long, it is interpreted as a literal list
      
    Args:
        fname (str): Path to the list file.
        root (str): Root directory for images.
        split (str): Dataset split (e.g., 'train' or 'val').
    
    Returns:
        list: List of tuples, each containing two full file paths.
    """
    if root and not root.endswith('/'):
        root += '/'
    assert os.path.isdir(root), f"Root directory does not exist: {root}"
    assert os.path.isfile(fname), f"File does not exist: {fname}"
    
    pairs = []
    root_split = os.path.join(root, split) + '/'
    with open(fname, 'r') as fid:
        for line in fid:
            image1_path, image2_path = line.strip().split(':')
            image2_path = image2_path[1:]
            pairs.append((os.path.join(root_split, image1_path),
                          os.path.join(root_split, image2_path)))
    return pairs

def dname_to_image_pairs(
    data_dir: str = 'data/ImageNet/',
    dname: str = 'ImageNet_preprocessed_dinoSmall',
    split: str = 'train',
    pairs_file_directory: str = 'pairs/'
) -> list:
    """
    Build image pairs from a dataset name and corresponding list file.
    
    Args:
        data_dir (str): Base directory for images.
        dname (str): Dataset name.
        split (str): Dataset split.
        pairs_file_directory (str): Directory containing the pairs list.
    
    Returns:
        list: List of image pair paths.
    """
    if not data_dir.endswith('/'):
        data_dir += '/'
    if not os.path.isdir(data_dir):
        print(f"Cannot find folder for {data_dir}.")    
    list_file = os.path.join(pairs_file_directory, dname) + split + '.txt'
    assert os.path.isfile(list_file), f"Cannot find list file for {dname} pairs: {list_file}"
    pairs = load_pairs_from_list_file(list_file, root=data_dir, split=split)
    print(f"  {dname}: {len(pairs):,} pairs")
    return pairs

# --- Dataset Class --- #

class PairsDataset(Dataset):
    def __init__(
        self,
        dname: str,
        split: str = 'train',
        data_dir_base: str = 'data/ImageNet/',
        pairs_file_directory: str = 'pairs/',
        image_size: int = 224,
        resize: int = 256,
        mask_dir: str = '',
        **kwargs
    ):
        """
        Initialize the pairs dataset.
        
        Args:
            dname (str): Dataset name.
            split (str): Dataset split.
            data_dir_base (str): Directory where images are stored.
            pairs_file_directory (str): Directory containing the list file for pairs.
            image_size (int): Size to center crop the images.
            crocodino_augmentations_ref (bool): Whether to apply extra augmentations for reference images.
            crocodino_augmentations_target (bool): Whether to apply extra augmentations for target images.
            resize (int): Size to resize images before cropping.
            mask_dir (str): Folder name for mask conversion.
        """
        super().__init__()
        self.image_pairs = dname_to_image_pairs(
            dname=dname, data_dir=data_dir_base, split=split, pairs_file_directory=pairs_file_directory
        )
        self.transforms = transforms.Compose([
            transforms.Resize(resize, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.mask_dir = mask_dir
        self.data_dir_base = data_dir_base

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, index: int):
        img1_path, img2_path = self.image_pairs[index]
        
        # Load the raw images.
        im1_loaded = load_img(img1_path)
        im2_loaded = load_img(img2_path)
        
        # Convert image paths to mask file paths.
        mask_img1_path = convert_mask_path(img1_path, self.mask_dir, data_dir_base=self.data_dir_base)
        mask_img2_path = convert_mask_path(img2_path, self.mask_dir, data_dir_base=self.data_dir_base)
        
        # Load the corresponding label masks.
        label1 = load_label(mask_img1_path)
        label2 = load_label(mask_img2_path)
        
        # Apply transforms to images.
        im1 = self.transforms(im1_loaded)

        im2 = self.transforms(im2_loaded)
    
        return im1, im2, label1, label2
