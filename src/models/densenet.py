"""
DenseNet-121 model for chest X-ray classification.
"""

import torch
import torch.nn as nn
from torchvision import models


class DenseNet121(nn.Module):
    """
    DenseNet-121 pretrained on ImageNet.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use pretrained weights from ImageNet
        dropout: Dropout probability for classifier head
    """
    
    def __init__(self, num_classes=5, pretrained=True, dropout=0.2):
        super(DenseNet121, self).__init__()
        
        # Load pretrained DenseNet-121
        if pretrained:
            self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        else:
            self.model = models.densenet121(weights=None)
        
        # Get number of input features for classifier
        num_features = self.model.classifier.in_features
        
        # Replace classifier head
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output logits (B, num_classes)
        """
        return self.model(x)
    
    def freeze_backbone(self):
        """Freeze all layers except classifier."""
        for param in self.model.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all layers."""
        for param in self.model.features.parameters():
            param.requires_grad = True


# Test the model
if __name__ == "__main__":
    print("Testing DenseNet-121 model...\n")
    
    # Test with pretrained weights
    print("1. Testing with pretrained weights...")
    model = DenseNet121(num_classes=5, pretrained=True, dropout=0.2)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test freezing/unfreezing
    print("\n2. Testing freeze/unfreeze...")
    model.freeze_backbone()
    frozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params (frozen): {frozen_trainable:,}")
    
    model.unfreeze_backbone()
    unfrozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params (unfrozen): {unfrozen_trainable:,}")
    
    print("\n✓ DenseNet-121 model test passed!")