# import the necessary packages
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

class SegmentationDataset(Dataset):

    def __init__(self, imagePaths, maskPaths, transforms=None, augmentations=None):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms
        self.augmentations = augmentations

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.imagePaths[idx]).convert('RGB')
        mask = Image.open(self.maskPaths[idx]).convert('RGB')

        # Apply augmentations if any
        if self.augmentations is not None:
            augmented = self.augmentations(image=np.array(image), mask=np.array(mask))
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])

        # Apply transforms if any
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        # Convert to tensor and ensure correct shape
        image = transforms.ToTensor()(image)  # Shape: (3, H, W)
        mask = transforms.ToTensor()(mask)    # Shape: (3, H, W)

        return image, mask