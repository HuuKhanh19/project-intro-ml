"""
Evaluation module for model assessment.
"""

from .metrics import (
    compute_metrics,
    evaluate_model,
    save_metrics,
    print_metrics
)
from .visualize import (
    plot_confusion_matrix,
    plot_training_curves,
    plot_roc_curves,
    save_classification_report
)

__all__ = [
    'compute_metrics',
    'evaluate_model',
    'save_metrics',
    'print_metrics',
    'plot_confusion_matrix',
    'plot_training_curves',
    'plot_roc_curves',
    'save_classification_report',
]