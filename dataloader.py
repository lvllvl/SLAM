import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import (
    BATCH_SIZE,
    IMG_HEIGHT,
    IMG_WIDTH,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    VAL_IMAGE_DIR,
    VAL_MASK_DIR,
)

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.png', '_mask.png'))
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Assuming mask is a grayscale image
        
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

# Define transformations
transform = transforms.Compose([
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),  # Resize images to a fixed size
    transforms.ToTensor(),  # Convert images and masks to PyTorch tensors
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)  # Normalize images
])

# Create dataset
train_dataset = SegmentationDataset(
    image_dir=TRAIN_IMAGE_DIR,
    mask_dir=TRAIN_MASK_DIR,
    transform=transform
)

val_dataset = SegmentationDataset(
    image_dir=VAL_IMAGE_DIR,
    mask_dir=VAL_MASK_DIR,
    transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Iterate over train_loader and val_loader during training and validation
def get_dataloaders(train_dir, train_maskdir, val_dir, val_maskdir, batch_size):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),  # Resize images to a fixed size
        transforms.ToTensor(),  # Convert images and masks to PyTorch tensors
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)  # Normalize images
    ])

    # Create datasets
    train_dataset = SegmentationDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=transform
    )

    val_dataset = SegmentationDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader