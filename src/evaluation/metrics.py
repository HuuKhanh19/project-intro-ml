"""
Metrics computation for model evaluation.
"""

import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm


def compute_metrics(y_true, y_pred, y_prob=None, class_names=None, average='macro'):
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        y_prob: Predicted probabilities (N, num_classes) - optional
        class_names: List of class names
        average: Averaging method for multi-class metrics
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # Per-class metrics
    metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    # AUC (if probabilities provided)
    if y_prob is not None:
        try:
            # One-vs-rest AUC for multi-class
            metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average=average)
            
            # Per-class AUC
            num_classes = y_prob.shape[1]
            auc_per_class = []
            for i in range(num_classes):
                y_true_binary = (y_true == i).astype(int)
                auc_per_class.append(roc_auc_score(y_true_binary, y_prob[:, i]))
            metrics['auc_per_class'] = auc_per_class
        except:
            metrics['auc'] = None
            metrics['auc_per_class'] = None
    
    # Classification report
    if class_names is not None:
        report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
        metrics['classification_report'] = report
    
    return metrics


def evaluate_model(model, dataloader, device, class_names=None):
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device to evaluate on
        class_names: List of class names
        
    Returns:
        Tuple of (metrics_dict, y_true, y_pred, y_prob)
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    
    return metrics, y_true, y_pred, y_prob


def save_metrics(metrics, save_path):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Metrics dictionary
        save_path: Path to save JSON file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    metrics_json = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_json[key] = value.tolist()
        elif isinstance(value, (np.int64, np.float64)):
            metrics_json[key] = value.item()
        else:
            metrics_json[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"Metrics saved to {save_path}")


def print_metrics(metrics, class_names=None):
    """
    Pretty print metrics.
    
    Args:
        metrics: Metrics dictionary
        class_names: List of class names
    """
    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)
    
    # Overall metrics
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall:    {metrics['recall']*100:.2f}%")
    print(f"  F1 Score:  {metrics['f1']*100:.2f}%")
    
    if metrics.get('auc') is not None:
        print(f"  AUC:       {metrics['auc']*100:.2f}%")
    
    # Per-class metrics
    if class_names is not None:
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
        print("-" * 70)
        
        for i, cls in enumerate(class_names):
            prec = metrics['precision_per_class'][i] * 100
            rec = metrics['recall_per_class'][i] * 100
            f1 = metrics['f1_per_class'][i] * 100
            
            line = f"{cls:<20} {prec:<12.2f} {rec:<12.2f} {f1:<12.2f}"
            
            if metrics.get('auc_per_class') is not None:
                auc = metrics['auc_per_class'][i] * 100
                line += f" {auc:<12.2f}"
            
            print(line)
    
    # Classification report
    if 'classification_report' in metrics:
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        print(metrics['classification_report'])
    
    print("="*70 + "\n")


# Test functions
if __name__ == "__main__":
    print("Testing metrics computation...\n")
    
    # Create dummy predictions
    np.random.seed(42)
    n_samples = 100
    n_classes = 5
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_prob = np.random.rand(n_samples, n_classes)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize
    
    class_names = ['Normal', 'Pneumonia', 'COVID', 'Tuberculosis', 'Pneumothorax']
    
    print("Test data:")
    print(f"  Samples: {n_samples}")
    print(f"  Classes: {n_classes}")
    print(f"  True labels shape: {y_true.shape}")
    print(f"  Predicted labels shape: {y_pred.shape}")
    print(f"  Probabilities shape: {y_prob.shape}\n")
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    
    # Print metrics
    print_metrics(metrics, class_names)
    
    # Test saving
    print("Testing save metrics...")
    save_metrics(metrics, 'test_metrics/metrics.json')
    
    # Cleanup
    import shutil
    shutil.rmtree('test_metrics')
    
    print("\n✓ All metrics tests passed!")