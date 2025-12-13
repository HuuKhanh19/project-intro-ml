"""
Evaluation script for trained models.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

from src.utils.config import load_config
from src.utils.logger import print_header, print_colored, Colors
from src.data import create_dataloaders
from src.models import create_model
from src.evaluation import (
    evaluate_model,
    save_metrics,
    print_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    save_classification_report
)


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained chest X-ray classification model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save results (default: same as checkpoint dir)')
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print_colored(f"ERROR: Checkpoint not found: {checkpoint_path}", Colors.BRIGHT_RED)
        sys.exit(1)
    
    # Load checkpoint
    print_header("CHEST X-RAY CLASSIFICATION - EVALUATION")
    print_colored(f"Loading checkpoint: {checkpoint_path}", Colors.BRIGHT_CYAN)
    
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']
    
    # Set save directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = checkpoint_path.parent / 'evaluation'
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_colored(f"Device: {device}\n", Colors.BRIGHT_BLUE)
    
    # Load data
    print_colored(f"Loading {args.split} data...", Colors.BRIGHT_CYAN)
    dataloaders = create_dataloaders(config)
    
    if args.split == 'train':
        dataloader = dataloaders['train']
    elif args.split == 'val':
        dataloader = dataloaders['val']
    else:
        dataloader = dataloaders['test']
    
    class_names = dataloaders['class_names']
    
    # Create model
    model_name = config['experiment']['model']
    print_colored(f"\nCreating model: {model_name}", Colors.BRIGHT_CYAN)
    model = create_model(model_name, config)
    model = model.to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print_colored("✓ Model weights loaded", Colors.BRIGHT_GREEN)
    
    # Evaluate
    print_colored(f"\nEvaluating on {args.split} set...", Colors.BRIGHT_CYAN)
    metrics, y_true, y_pred, y_prob = evaluate_model(
        model,
        dataloader,
        device,
        class_names
    )
    
    # Print metrics
    print_metrics(metrics, class_names)
    
    # Save metrics
    print_colored("\nSaving results...", Colors.BRIGHT_CYAN)
    metrics_path = save_dir / f'metrics_{args.split}.json'
    save_metrics(metrics, metrics_path)
    
    # Plot confusion matrix
    cm_path = save_dir / f'confusion_matrix_{args.split}.png'
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    
    # Plot normalized confusion matrix
    cm_norm_path = save_dir / f'confusion_matrix_{args.split}_normalized.png'
    plot_confusion_matrix(y_true, y_pred, class_names, cm_norm_path, normalize=True)
    
    # Plot ROC curves
    roc_path = save_dir / f'roc_curves_{args.split}.png'
    plot_roc_curves(y_true, y_prob, class_names, roc_path)
    
    # Save classification report
    report_path = save_dir / f'classification_report_{args.split}.txt'
    save_classification_report(metrics, report_path)
    
    # Summary
    print_header("EVALUATION COMPLETED")
    print_colored(f"✓ Results saved to: {save_dir}", Colors.BRIGHT_GREEN, bold=True)
    print_colored(f"  - Metrics: {metrics_path.name}", Colors.BRIGHT_BLUE)
    print_colored(f"  - Confusion matrix: {cm_path.name}", Colors.BRIGHT_BLUE)
    print_colored(f"  - ROC curves: {roc_path.name}", Colors.BRIGHT_BLUE)
    print_colored(f"  - Classification report: {report_path.name}\n", Colors.BRIGHT_BLUE)


if __name__ == "__main__":
    main()