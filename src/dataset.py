"""
Dataset class for chest X-ray classification with medical-specific augmentation.
- Runtime augmentation for minority classes (COVID, Tuberculosis, Pneumothorax)
- Medical-appropriate augmentations: Horizontal Flip, Rotation, Gamma, CLAHE
- ImageNet normalization for pretrained models
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class ChestXrayDataset(Dataset):
    """
    Chest X-ray Dataset with class-specific augmentation.
    
    Args:
        data_dir (str): Path to data directory (e.g., 'data/processed/train')
        transform (callable, optional): Albumentations transform pipeline
        augment_minority (bool): Whether to apply augmentation to minority classes
    """
    
    def __init__(self, data_dir, transform=None, augment_minority=False):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.augment_minority = augment_minority
        
        # Class names and labels
        self.classes = ['Normal', 'Pneumonia', 'COVID', 'Tuberculosis', 'Pneumothorax']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Minority classes that need augmentation
        self.minority_classes = {'COVID', 'Tuberculosis', 'Pneumothorax'}
        
        # Load all image paths and labels
        self.samples = []
        self._load_data()
        
        print(f"Loaded {len(self.samples)} images from {self.data_dir}")
        self._print_class_distribution()
    
    def _load_data(self):
        """Load all image paths and their labels."""
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist, skipping...")
                continue
            
            # Get all PNG images
            image_files = sorted(class_dir.glob("*.png"))
            
            for img_path in image_files:
                self.samples.append({
                    'path': img_path,
                    'label': self.class_to_idx[class_name],
                    'class_name': class_name
                })
    
    def _print_class_distribution(self):
        """Print class distribution in the dataset."""
        class_counts = {cls: 0 for cls in self.classes}
        
        for sample in self.samples:
            class_counts[sample['class_name']] += 1
        
        print("\nClass distribution:")
        for cls, count in class_counts.items():
            aug_status = "(with augmentation)" if cls in self.minority_classes and self.augment_minority else ""
            print(f"  {cls:15s}: {count:5d} images {aug_status}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get image and label at index.
        
        Returns:
            image (torch.Tensor): Preprocessed image tensor (C, H, W)
            label (int): Class label
        """
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['path']).convert('RGB')
        image = np.array(image)
        
        # Apply augmentation only to minority classes during training
        if self.augment_minority and sample['class_name'] in self.minority_classes:
            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed['image']
        else:
            # Apply basic transform (normalization only)
            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed['image']
        
        label = sample['label']
        
        return image, label
    
    def get_class_weights(self):
        """
        Calculate class weights for weighted loss (inverse frequency).
        
        Returns:
            torch.Tensor: Class weights
        """
        class_counts = [0] * len(self.classes)
        
        for sample in self.samples:
            class_counts[sample['label']] += 1
        
        # Calculate weights: inverse frequency
        total_samples = len(self.samples)
        class_weights = [total_samples / (len(self.classes) * count) for count in class_counts]
        
        return torch.FloatTensor(class_weights)


def get_train_transform(image_size=224):
    """
    Get training transform with medical-appropriate augmentation for minority classes.
    
    Augmentations:
    - Horizontal Flip: Lungs are relatively symmetric
    - Random Rotation: ±5-10°, patients not always perfectly aligned
    - Gamma Correction: Simulate different X-ray exposure levels
    - CLAHE: Enhance local contrast for bone/tissue structures
    """
    return A.Compose([
        # Geometric augmentation
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.3),
        
        # Photometric augmentation
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),  # Simulate different X-ray exposures
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),  # Enhance local contrast
        
        # Normalization (ImageNet stats for pretrained models)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


def get_val_transform(image_size=224):
    """
    Get validation/test transform (no augmentation, only normalization).
    """
    return A.Compose([
        # Only normalization, NO augmentation for fair evaluation
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


def create_dataloaders(data_dir='data/processed', 
                      batch_size=32, 
                      num_workers=4,
                      image_size=224):
    """
    Create train, validation and test dataloaders.
    
    Args:
        data_dir (str): Path to processed data directory
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        image_size (int): Image size (assumes square images)
    
    Returns:
        dict: Dictionary containing train_loader, val_loader, test_loader
    """
    data_dir = Path(data_dir)
    
    # Create datasets
    train_dataset = ChestXrayDataset(
        data_dir=data_dir / 'train',
        transform=get_train_transform(image_size),
        augment_minority=True  # Apply augmentation to minority classes
    )
    
    val_dataset = ChestXrayDataset(
        data_dir=data_dir / 'val',
        transform=get_val_transform(image_size),
        augment_minority=False  # No augmentation for validation
    )
    
    test_dataset = ChestXrayDataset(
        data_dir=data_dir / 'test',
        transform=get_val_transform(image_size),
        augment_minority=False  # No augmentation for test
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop incomplete batch for stable batch norm
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Get class weights for weighted loss
    class_weights = train_dataset.get_class_weights()
    
    print("\n" + "="*60)
    print("DATALOADER SUMMARY")
    print("="*60)
    print(f"Train batches: {len(train_loader)} (batch_size={batch_size})")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    print("\nClass weights (for weighted loss):")
    for i, cls in enumerate(train_dataset.classes):
        print(f"  {cls:15s}: {class_weights[i]:.4f}")
    print("="*60 + "\n")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'class_weights': class_weights,
        'classes': train_dataset.classes
    }


# Test the dataset
if __name__ == "__main__":
    print("Testing ChestXrayDataset...\n")
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        data_dir='data/processed',
        batch_size=16,
        num_workers=2
    )
    
    # Test one batch
    train_loader = dataloaders['train']
    images, labels = next(iter(train_loader))
    
    print("\nSample batch:")
    print(f"  Images shape: {images.shape}")  # (B, C, H, W)
    print(f"  Labels shape: {labels.shape}")  # (B,)
    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Label values: {labels.unique().tolist()}")
    
    # Check class distribution in one batch
    print("\nClass distribution in sample batch:")
    classes = dataloaders['classes']
    for i, cls in enumerate(classes):
        count = (labels == i).sum().item()
        print(f"  {cls:15s}: {count} images")
    
    print("\n✓ Dataset test completed successfully!")