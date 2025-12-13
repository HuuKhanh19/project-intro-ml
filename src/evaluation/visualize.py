"""
Visualization functions for evaluation results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, normalize=False):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save plot
        normalize: Normalize confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_training_curves(history, save_path=None, model_name=None, loss_name=None):
    """
    Plot training and validation curves.
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot
        model_name: Model name for title
        loss_name: Loss name for title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    
    title = 'Loss Curves'
    if model_name and loss_name:
        title = f'{model_name} - {loss_name}\n{title}'
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    
    title = 'Accuracy Curves'
    if model_name and loss_name:
        title = f'{model_name} - {loss_name}\n{title}'
    ax2.set_title(title, fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Add best epoch marker
    best_epoch = np.argmax(history['val_acc']) + 1
    best_acc = history['val_acc'][best_epoch - 1]
    ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
    ax2.plot(best_epoch, best_acc, 'g*', markersize=15, label=f'Best: {best_acc:.2f}%')
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.close()


def plot_roc_curves(y_true, y_prob, class_names, save_path=None):
    """
    Plot ROC curves for each class.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities (N, num_classes)
        class_names: List of class names
        save_path: Path to save plot
    """
    n_classes = len(class_names)
    
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        # Binary labels for this class
        y_true_binary = (y_true == i).astype(int)
        y_score = y_prob[:, i]
        
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, color=color, lw=2,
                label=f'{cls} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - One-vs-Rest', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    plt.close()


def save_classification_report(metrics, save_path):
    """
    Save classification report to text file.
    
    Args:
        metrics: Metrics dictionary containing classification_report
        save_path: Path to save text file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        if 'classification_report' in metrics:
            f.write(metrics['classification_report'])
        
        f.write("\n" + "="*70 + "\n")
        f.write("SUMMARY METRICS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Accuracy:  {metrics['accuracy']*100:.2f}%\n")
        f.write(f"Precision: {metrics['precision']*100:.2f}%\n")
        f.write(f"Recall:    {metrics['recall']*100:.2f}%\n")
        f.write(f"F1 Score:  {metrics['f1']*100:.2f}%\n")
        
        if metrics.get('auc') is not None:
            f.write(f"AUC:       {metrics['auc']*100:.2f}%\n")
    
    print(f"Classification report saved to {save_path}")


# Test functions
if __name__ == "__main__":
    print("Testing visualization functions...\n")
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 100
    n_classes = 5
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_prob = np.random.rand(n_samples, n_classes)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    class_names = ['Normal', 'Pneumonia', 'COVID', 'Tuberculosis', 'Pneumothorax']
    
    # Test confusion matrix
    print("1. Testing confusion matrix plot...")
    plot_confusion_matrix(y_true, y_pred, class_names, 'test_viz/confusion_matrix.png')
    
    # Test training curves
    print("2. Testing training curves plot...")
    history = {
        'train_loss': [1.5, 1.2, 1.0, 0.8, 0.7],
        'val_loss': [1.4, 1.3, 1.1, 0.9, 0.85],
        'train_acc': [50, 60, 70, 80, 85],
        'val_acc': [48, 58, 68, 78, 82]
    }
    plot_training_curves(history, 'test_viz/training_curves.png', 'DenseNet121', 'Focal Loss')
    
    # Test ROC curves
    print("3. Testing ROC curves plot...")
    plot_roc_curves(y_true, y_prob, class_names, 'test_viz/roc_curves.png')
    
    # Test classification report
    print("4. Testing classification report save...")
    from .metrics import compute_metrics
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    save_classification_report(metrics, 'test_viz/classification_report.txt')
    
    # Cleanup
    import shutil
    shutil.rmtree('test_viz')
    
    print("\n✓ All visualization tests passed!")