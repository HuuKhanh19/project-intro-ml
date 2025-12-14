"""
Chest X-ray Classification Package

A comprehensive deep learning framework for classifying chest X-ray images
into 5 disease categories: Normal, Pneumonia, COVID-19, Tuberculosis, and Pneumothorax.

Modules:
    - data: Dataset and data loading utilities
    - models: Model architectures (MLP, LeNet, DenseNet-121, EfficientNet-B0)
    - training: Training loop and utilities
    - evaluation: Metrics computation and visualization
    - utils: Configuration and logging utilities
"""

__version__ = "1.0.0"
__author__ = "Huu Khanh Dang"

# Import main components
from . import data
from . import models
from . import training
from . import evaluation
from . import utils

__all__ = [
    'data',
    'models',
    'training',
    'evaluation',
    'utils',
]