"""
Model architectures and loss functions.
"""

from .mlp import MLP
from .lenet import LeNet
from .densenet import DenseNet121
from .efficientnet import EfficientNetB0
from .losses import WeightedCELoss, FocalLoss


def create_model(model_name, config):
    """
    Factory function to create model based on name.
    
    Args:
        model_name: Model name (mlp, lenet, densenet121, efficientnet_b0)
        config: Configuration dictionary
        
    Returns:
        Model instance
    """
    num_classes = config['project']['num_classes']
    model_config = config.get('model_config', {})
    
    model_name = model_name.lower()
    
    if model_name == 'mlp':
        return MLP(
            num_classes=num_classes,
            dropout=model_config.get('dropout', 0.5),
            hidden_dims=model_config.get('hidden_dims', [1024, 512])
        )
    elif model_name == 'lenet':
        return LeNet(
            num_classes=num_classes,
            dropout=model_config.get('dropout', 0.5)
        )
    elif model_name == 'densenet121':
        return DenseNet121(
            num_classes=num_classes,
            pretrained=model_config.get('pretrained', True),
            dropout=model_config.get('dropout', 0.2)
        )
    elif model_name == 'efficientnet_b0':
        return EfficientNetB0(
            num_classes=num_classes,
            pretrained=model_config.get('pretrained', True),
            dropout=model_config.get('dropout', 0.2)
        )
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from ['mlp', 'lenet', 'densenet121', 'efficientnet_b0']"
        )


def create_loss(loss_name, class_weights=None, config=None):
    """
    Factory function to create loss function.
    
    Args:
        loss_name: Loss name (weighted_ce, focal)
        class_weights: Class weights tensor
        config: Configuration dictionary (optional)
        
    Returns:
        Loss function instance
    """
    loss_name = loss_name.lower()
    
    if loss_name == 'weighted_ce':
        return WeightedCELoss(weight=class_weights)
    elif loss_name == 'focal':
        gamma = 2.0
        if config and 'loss' in config and 'focal' in config['loss']:
            gamma = config['loss']['focal'].get('gamma', 2.0)
        return FocalLoss(alpha=class_weights, gamma=gamma)
    else:
        raise ValueError(
            f"Unknown loss: {loss_name}. "
            f"Choose from ['weighted_ce', 'focal']"
        )


def count_parameters(model):
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    'MLP',
    'LeNet',
    'DenseNet121',
    'EfficientNetB0',
    'WeightedCELoss',
    'FocalLoss',
    'create_model',
    'create_loss',
    'count_parameters',
]