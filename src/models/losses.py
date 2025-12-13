"""
Loss functions for chest X-ray classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCELoss(nn.Module):
    """
    Weighted Cross-Entropy Loss with class weights.
    
    Args:
        weight: Class weights tensor (num_classes,)
    """
    
    def __init__(self, weight=None):
        super(WeightedCELoss, self).__init__()
        self.weight = weight
    
    def forward(self, inputs, targets):
        """
        Forward pass.
        
        Args:
            inputs: Model predictions (B, num_classes) - logits
            targets: Ground truth labels (B,)
            
        Returns:
            Loss value (scalar)
        """
        return F.cross_entropy(inputs, targets, weight=self.weight)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Class weights tensor (num_classes,) or None
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('mean', 'sum', 'none')
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
        https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Forward pass.
        
        Args:
            inputs: Model predictions (B, num_classes) - logits
            targets: Ground truth labels (B,)
            
        Returns:
            Loss value (scalar or tensor depending on reduction)
        """
        # Get softmax probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get cross-entropy loss (without reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get class probabilities for target classes
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Calculate focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal weight to cross-entropy loss
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            # Get alpha values for target classes
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Test loss functions
if __name__ == "__main__":
    print("Testing loss functions...\n")
    
    # Create dummy data
    batch_size = 8
    num_classes = 5
    
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Class weights (inverse frequency example)
    class_weights = torch.tensor([0.5, 0.8, 1.3, 1.9, 1.7])
    
    print("Test data:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Targets: {targets.tolist()}")
    print(f"  Class weights: {class_weights.tolist()}\n")
    
    # Test Weighted CE Loss
    print("1. Testing Weighted Cross-Entropy Loss...")
    wce_loss = WeightedCELoss(weight=class_weights)
    loss_value = wce_loss(logits, targets)
    print(f"   Loss value: {loss_value.item():.4f}")
    
    # Compare with standard CE (no weights)
    standard_ce = F.cross_entropy(logits, targets)
    print(f"   Standard CE (no weights): {standard_ce.item():.4f}")
    print(f"   Difference: {abs(loss_value.item() - standard_ce.item()):.4f}\n")
    
    # Test Focal Loss
    print("2. Testing Focal Loss (gamma=2.0)...")
    focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
    loss_value = focal_loss(logits, targets)
    print(f"   Loss value: {loss_value.item():.4f}")
    
    # Test different gamma values
    print("\n3. Testing different gamma values...")
    for gamma in [0.0, 1.0, 2.0, 3.0]:
        fl = FocalLoss(alpha=class_weights, gamma=gamma)
        loss_val = fl(logits, targets)
        print(f"   Gamma={gamma:.1f}: {loss_val.item():.4f}")
    
    print("\n4. Testing without alpha (class weights)...")
    fl_no_alpha = FocalLoss(alpha=None, gamma=2.0)
    loss_value = fl_no_alpha(logits, targets)
    print(f"   Loss value: {loss_value.item():.4f}")
    
    # Test reduction methods
    print("\n5. Testing reduction methods...")
    fl = FocalLoss(alpha=class_weights, gamma=2.0, reduction='none')
    loss_per_sample = fl(logits, targets)
    print(f"   No reduction shape: {loss_per_sample.shape}")
    print(f"   Per-sample losses: {loss_per_sample.tolist()}")
    
    fl_sum = FocalLoss(alpha=class_weights, gamma=2.0, reduction='sum')
    loss_sum = fl_sum(logits, targets)
    print(f"   Sum reduction: {loss_sum.item():.4f}")
    
    print("\n✓ All loss function tests passed!")