"""
EfficientNet-B0 model for chest X-ray classification.
"""

import torch
import torch.nn as nn
import timm


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 pretrained on ImageNet using timm library.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use pretrained weights from ImageNet
        dropout: Dropout probability for classifier head
    """
    
    def __init__(self, num_classes=5, pretrained=True, dropout=0.2):
        super(EfficientNetB0, self).__init__()
        
        # Load EfficientNet-B0 from timm
        if pretrained:
            self.model = timm.create_model(
                'efficientnet_b0',
                pretrained=True,
                num_classes=num_classes,
                drop_rate=dropout
            )
        else:
            self.model = timm.create_model(
                'efficientnet_b0',
                pretrained=False,
                num_classes=num_classes,
                drop_rate=dropout
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
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier
        for param in self.model.classifier.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        """Unfreeze all layers."""
        for param in self.model.parameters():
            param.requires_grad = True


# Test the model
if __name__ == "__main__":
    print("Testing EfficientNet-B0 model...\n")
    
    # Test with pretrained weights
    print("1. Testing with pretrained weights...")
    model = EfficientNetB0(num_classes=5, pretrained=True, dropout=0.2)
    
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
    
    print("\n✓ EfficientNet-B0 model test passed!")