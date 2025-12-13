"""
Modern LeNet architecture adapted for chest X-ray classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """
    Modern LeNet with BatchNorm and Dropout.
    Adapted for 224x224 RGB images.
    
    Architecture:
        Conv Block 1: Conv(3→32) → BN → ReLU → MaxPool (224→112)
        Conv Block 2: Conv(32→64) → BN → ReLU → MaxPool (112→56)
        Conv Block 3: Conv(64→128) → BN → ReLU → MaxPool (56→28)
        Conv Block 4: Conv(128→256) → BN → ReLU → MaxPool (28→14)
        FC Block: FC(256*14*14→512) → Dropout → FC(512→256) → Dropout → FC(256→num_classes)
    
    Args:
        num_classes: Number of output classes
        dropout: Dropout probability
    """
    
    def __init__(self, num_classes=5, dropout=0.5):
        super(LeNet, self).__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 224 → 112
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 112 → 56
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 56 → 28
        
        # Conv Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  # 28 → 14
        
        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output logits (B, num_classes)
        """
        # Conv blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # FC layers
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x


# Test the model
if __name__ == "__main__":
    print("Testing LeNet model...\n")
    
    # Create model
    model = LeNet(num_classes=5, dropout=0.5)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    print("\n✓ LeNet model test passed!")