"""
4 models for comparison (FINAL - optimized for small medical dataset):
1. MLP (baseline - worst expected)
2. LeNet-5 (CNN basic)
3. ResNet-18 (CNN advanced - smaller & less overfit than ResNet50)
4. EfficientNet-B0 (SOTA - smallest EfficientNet for small datasets)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, EfficientNet_B0_Weights
import yaml


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron"""
    def __init__(self, input_size=224, num_classes=5, hidden_dims=[512, 256, 128], dropout=0.6):
        super(MLP, self).__init__()
        
        flatten_size = input_size * input_size * 3
        
        layers = []
        in_features = flatten_size
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_features = hidden_dim
        
        layers.append(nn.Linear(in_features, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.network(x)


class LeNet5(nn.Module):
    """LeNet-5 adapted for chest X-rays"""
    def __init__(self, num_classes=5, dropout=0.5):
        super(LeNet5, self).__init__()
        
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv2
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16 * 54 * 54, 120),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(84, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet18Custom(nn.Module):
    """ResNet-18 - Smaller than ResNet50, less prone to overfitting"""
    def __init__(self, num_classes=5, pretrained=True, dropout=0.6):
        super(ResNet18Custom, self).__init__()
        
        if pretrained:
            self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet18(weights=None)
        
        # Replace final FC layer with dropout
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)


class EfficientNetB0Custom(nn.Module):
    """EfficientNet-B0 - Smallest EfficientNet, optimized for small datasets"""
    def __init__(self, num_classes=5, pretrained=True, dropout=0.5):
        super(EfficientNetB0Custom, self).__init__()
        
        if pretrained:
            self.efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.efficientnet = models.efficientnet_b0(weights=None)
        
        # Replace classifier
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.efficientnet(x)


def get_model(model_name, config_path='configs/config.yaml'):
    """
    Factory function to get model by name
    
    Args:
        model_name: 'mlp', 'lenet', 'resnet18', 'efficientnet_b0'
        config_path: Path to config file
    
    Returns:
        model: PyTorch model
    """
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    num_classes = config['data']['num_classes']
    image_size = config['data']['image_size']
    
    if model_name == 'mlp':
        model_config = config['models']['mlp']
        model = MLP(
            input_size=image_size,
            num_classes=num_classes,
            hidden_dims=model_config['hidden_dims'],
            dropout=model_config['dropout']
        )
    
    elif model_name == 'lenet':
        model_config = config['models']['lenet']
        model = LeNet5(
            num_classes=num_classes,
            dropout=model_config['dropout']
        )
    
    elif model_name == 'resnet18':
        model_config = config['models']['resnet18']
        model = ResNet18Custom(
            num_classes=num_classes,
            pretrained=model_config['pretrained'],
            dropout=model_config['dropout']
        )
    
    elif model_name == 'efficientnet_b0':
        model_config = config['models']['efficientnet_b0']
        model = EfficientNetB0Custom(
            num_classes=num_classes,
            pretrained=model_config['pretrained'],
            dropout=model_config['dropout']
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test all models
    models_list = ['mlp', 'lenet', 'resnet18', 'efficientnet_b0']
    
    print("=" * 80)
    print("MODEL ARCHITECTURES - FINAL")
    print("=" * 80)
    
    for model_name in models_list:
        model = get_model(model_name)
        params = count_parameters(model)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        
        print(f"\n{model_name.upper()}:")
        print(f"  Parameters: {params:,}")
        print(f"  Output shape: {output.shape}")
    
    print("\n" + "=" * 80)
    print("All models ready for training!")
    print("=" * 80)