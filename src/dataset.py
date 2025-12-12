"""
Dataset class with smart augmentation strategy
Strong augmentation for minority classes (TB, Pneumothorax)
Mild augmentation for majority classes (Normal, Pneumonia, COVID)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import yaml

class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, split='train', config_path='configs/config.yaml'):
        """
        Args:
            root_dir: Path to processed data (e.g., 'data/processed')
            split: 'train', 'val', or 'test'
            config_path: Path to config.yaml
        """
        self.root_dir = Path(root_dir) / split
        self.split = split
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.class_names = self.config['data']['class_names']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Minority classes (need strong augmentation)
        self.minority_classes = ['Tuberculosis', 'Pneumothorax']
        
        # Load image paths
        self.samples = []
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.*'):
                    self.samples.append((img_path, self.class_to_idx[class_name], class_name))
        
        # Setup transforms
        self.transform = self._get_transforms()
    
    def _get_transforms(self):
        """Get transforms based on split and class - Medical imaging appropriate"""
        image_size = self.config['data']['image_size']
        
        if self.split == 'train':
            # Medical imaging appropriate augmentation
            return {
                'base': transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'strong': transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomRotation(15),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomAffine(
                        degrees=0, 
                        translate=(0.05, 0.05),
                        scale=(0.95, 1.05)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'mild': transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomRotation(10),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(0.03, 0.03),
                        scale=(0.97, 1.03)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            }
        else:
            return {
                'base': transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, class_name = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply class-specific augmentation
        if self.split == 'train':
            if class_name in self.minority_classes:
                image = self.transform['strong'](image)
            else:
                image = self.transform['mild'](image)
        else:
            image = self.transform['base'](image)
        
        return image, label


def get_dataloaders(config_path='configs/config.yaml'):
    """Create train, val, test dataloaders"""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_dir = config['data']['processed_dir']
    batch_size = config['training']['batch_size']
    
    # Create datasets
    train_dataset = ChestXrayDataset(data_dir, split='train', config_path=config_path)
    val_dataset = ChestXrayDataset(data_dir, split='val', config_path=config_path)
    test_dataset = ChestXrayDataset(data_dir, split='test', config_path=config_path)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset loaded:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample labels: {labels[:5]}")