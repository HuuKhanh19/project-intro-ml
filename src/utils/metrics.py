import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, test_loader, device, class_names):
    """
    Đánh giá model trên test set
    
    Returns:
        y_true: ground truth labels
        y_pred: predicted labels
        y_probs: prediction probabilities
    """
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
    
    return np.array(y_true), np.array(y_pred), np.array(y_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm


def print_classification_report(y_true, y_pred, class_names):
    """Print detailed classification report"""
    report = classification_report(y_true, y_pred, 
                                   target_names=class_names,
                                   digits=4)
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(report)
    
    return report


def calculate_per_class_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    print("\n" + "=" * 70)
    print("PER-CLASS ACCURACY")
    print("=" * 70)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:15s}: {per_class_acc[i]*100:.2f}%")
    
    return per_class_acc