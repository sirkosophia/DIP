import os
import ast
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torchvision.transforms as T


def load_img(file: str) -> Image.Image:
    """
    Load an image file and ensure it is in RGB mode.
    """
    image = Image.open(file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def load_label(file: str) -> torch.Tensor:
    """
    Load a label tensor from a file.
    """
    return torch.load(file)


def convert_to_mask_path(img_path: str, mask_dir: str, data_dir_base: str) -> str:
    """
    Convert an image path to the corresponding mask file path.
    Replaces the dataset base path and file extension.

    Args:
        img_path (str): Original image path.
        mask_dir (str): Folder suffix to add to the final mask directory.
        data_dir_base (str): Base directory for the dataset.

    Returns:
        str: Converted mask file path.
    """
    new_path = img_path.replace(data_dir_base, mask_dir).replace('.jpg', '.pt')
    return new_path



def load_pairs_from_list_file(fname: str, root: str = '', split: str = "train") -> List[Tuple[str, str]]:
    """
    Load image pairs from a text file.
    
    Each line should be formatted as 'image1_path:image2_paths'.
    For COCO, the split is appended with "2017". The second image path is assumed
    to have an extra leading character which is removed.
    
    Args:
        fname (str): Path to the list file.
        root (str): Root directory for images.
        split (str): Dataset split.
    
    Returns:
        List[Tuple[str, str]]: List of tuples with full paths for image pairs.
    """
    if root and not root.endswith('/'):
        root += '/'
    assert os.path.isdir(root), f"Root directory does not exist: {root}"
    assert os.path.isfile(fname), f"Cannot parse pairs from {fname}, file does not exist"

    if "COCO" in fname:
        split += "2017"

    # Construct the root path for this split.
    root_split = os.path.join(root, split) + '/'
    pairs = []
    with open(fname, 'r') as fid:
        for line in fid:
            image1_path, image2_paths = line.strip().split(':')
            image2_path = image2_paths[1:]  # Remove the first character
            pairs.append((os.path.join(root_split, image1_path),
                          os.path.join(root_split, image2_path)))
    return pairs


def dname_to_image_pairs(data_dir: str = 'data/COCO/images/',
                         dname: str = 'COCO_diffcut_k1000_normnonorm',
                         split: str = 'train',
                         pairs_file_directory: str = 'pairs/') -> List[Tuple[str, str]]:
    """
    Generate image pairs for a given dataset name.
    Args:
        data_dir (str): Directory containing dataset images.
        dname (str): Dataset name.
        split (str): Dataset split (e.g., 'train', 'val').
        pairs_file_directory (str): Directory containing the pairs list file.
    
    Returns:
        List[Tuple[str, str]]: List of image pair paths.
    """
    if not data_dir.endswith('/'):
        data_dir += '/'
    if not os.path.isdir(data_dir):
        print(f"Cannot find folder for {data_dir}")
    list_file = os.path.join(pairs_file_directory, dname) + split + '.txt'
    assert os.path.isfile(list_file), f"Cannot find list file for {dname} pairs: {list_file}"
    pairs = load_pairs_from_list_file(list_file, root=data_dir, split=split)
    print(f"  {dname}: {len(pairs):,} pairs")
    return pairs


class PairsDataset(Dataset):
    def __init__(self, 
                 dname: str,
                 split: str = 'train',
                 data_dir_base: str = 'data/',
                 pairs_file_directory: str = 'pairs/',
                 image_size: int = 224,
                 resize: int = 256,
                 mask_dir: str = '',
                 backbone_type: str = "dino",
                 **kwargs):
        """
        Initialize the dataset.
        Args:
            dname (str): Dataset name.
            split (str): Dataset split.
            data_dir_base (str): Directory where images are stored.
            pairs_file_directory (str): Directory where the pairs list file is located.
            image_size (int): Target size for center cropping.
            resize (int): Size to resize images.
            mask_dir (str): Folder to use for mask file conversion.
            backbone_type (str): Determines which transforms to use (e.g., "dino", "mae", or "clip").
        """
        super().__init__()
        # Adjust resize if using larger image dimensions.
        if image_size == 448:
            resize = 512

        # Append images subdirectory.
        data_dir = os.path.join(data_dir_base, 'COCO/', 'images/')
        print("Data directory:", data_dir)
        self.image_pairs = dname_to_image_pairs(dname=dname, data_dir=data_dir, split=split, pairs_file_directory=pairs_file_directory)

        # Select image transforms based on backbone type.
        if backbone_type in ("dino", "mae"):
            self.transforms = transforms.Compose([
                transforms.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")

        # Transform for label masks (using nearest neighbor interpolation).
        self.transform_label = transforms.Compose([
            transforms.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
        ])

        self.mask_dir = mask_dir
        self.data_dir_base = data_dir_base

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, index: int):
        img1_path, img2_path = self.image_pairs[index]

        # Load images.
        im1_loaded = load_img(img1_path)
        im2_loaded = load_img(img2_path)

        # Convert image paths to mask paths.
        mask_img1_path = convert_to_mask_path(img1_path, self.mask_dir, data_dir_base=self.data_dir_base)
        mask_img2_path = convert_to_mask_path(img2_path, self.mask_dir, data_dir_base=self.data_dir_base)

        # Load labels and convert them to long tensors.
        label1 = load_label(mask_img1_path).long()
        label2 = load_label(mask_img2_path).long()

        # Transform labels.
        label1 = self.transform_label(label1)
        label2 = self.transform_label(label2)

        # Apply image transforms.
        im1 = self.transforms(im1_loaded)
        im2 = self.transforms(im2_loaded)

        return im1, im2, label1, label2


