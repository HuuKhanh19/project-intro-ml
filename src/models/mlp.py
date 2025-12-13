"""
Multi-Layer Perceptron (MLP) baseline model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-Layer Perceptron baseline.
    
    Architecture: Flatten → FC layers with BatchNorm & Dropout → Output
    Default: [150528 → 1024 → 512 → num_classes]
    
    Args:
        num_classes: Number of output classes
        input_size: Input image size (assumes square images)
        dropout: Dropout probability
        hidden_dims: List of hidden layer dimensions
    """
    
    def __init__(self, num_classes=5, input_size=224, dropout=0.5, hidden_dims=None):
        super(MLP, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [1024, 512]
        
        # Calculate flattened input size: H * W * C
        self.input_features = input_size * input_size * 3
        
        self.flatten = nn.Flatten()
        
        # Build MLP layers dynamically
        layers = []
        in_features = self.input_features
        
        for i, out_features in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_features = out_features
        
        # Output layer
        layers.append(nn.Linear(in_features, num_classes))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output logits (B, num_classes)
        """
        x = self.flatten(x)
        x = self.mlp(x)
        return x


# Test the model
if __name__ == "__main__":
    print("Testing MLP model...\n")
    
    # Create model
    model = MLP(num_classes=5, dropout=0.5, hidden_dims=[1024, 512])
    
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
    
    print("\n✓ MLP model test passed!")