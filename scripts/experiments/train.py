"""
Main training script for chest X-ray classification experiments.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np

from src.utils.config import get_experiment_config, get_model_and_loss, save_config, print_config
from src.utils.logger import setup_logger, print_header, print_colored, Colors
from src.data import create_dataloaders
from src.models import create_model, create_loss, count_parameters
from src.training import Trainer, create_optimizer, create_scheduler
from src.evaluation import plot_training_curves


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Train chest X-ray classification model')
    
    # Experiment selection (choose one)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--experiment', type=str,
                       help='Experiment ID (e.g., exp01_densenet121_weighted_ce)')
    group.add_argument('--model', type=str,
                       choices=['mlp', 'lenet', 'densenet121', 'efficientnet_b0'],
                       help='Model name (used with --loss)')
    
    # Loss function (required if using --model)
    parser.add_argument('--loss', type=str,
                        choices=['weighted_ce', 'focal'],
                        help='Loss function (required if using --model)')
    
    # Optional overrides
    parser.add_argument('--epochs', type=int, help='Number of epochs (override config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (override config)')
    parser.add_argument('--lr', type=float, help='Learning rate (override config)')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.model and not args.loss:
        parser.error("--loss is required when using --model")
    
    # Load configuration
    if args.experiment:
        config = get_experiment_config(args.experiment)
    else:
        config = get_model_and_loss(args.model, args.loss)
    
    # Apply overrides
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['optimizer']['lr'] = args.lr
    
    # Set random seed
    set_seed(config['project']['seed'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Print header
    print_header("CHEST X-RAY CLASSIFICATION - TRAINING")
    print_config(config)
    
    print_colored(f"Device: {device}", Colors.BRIGHT_BLUE)
    if device.type == 'cuda':
        print_colored(f"GPU: {torch.cuda.get_device_name(0)}", Colors.BRIGHT_BLUE)
        print_colored(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n", Colors.BRIGHT_BLUE)
    
    # Create dataloaders
    print_colored("\nLoading data...", Colors.BRIGHT_CYAN)
    dataloaders = create_dataloaders(config)
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    class_weights = dataloaders['class_weights'].to(device)
    
    # Create model
    model_name = config['experiment']['model']
    print_colored(f"\nCreating model: {model_name}", Colors.BRIGHT_CYAN)
    model = create_model(model_name, config)
    model = model.to(device)
    
    print_colored(f"Parameters: {count_parameters(model):,}", Colors.BRIGHT_GREEN)
    
    # Create loss function
    loss_name = config['experiment']['loss']
    print_colored(f"\nCreating loss function: {loss_name}", Colors.BRIGHT_CYAN)
    criterion = create_loss(loss_name, class_weights, config)
    
    # Create optimizer
    print_colored("\nSetting up optimizer and scheduler...", Colors.BRIGHT_CYAN)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    opt_config = config['training']['optimizer']
    print_colored(f"Optimizer: {opt_config['type']} (lr={opt_config['lr']}, wd={opt_config['weight_decay']})", Colors.BRIGHT_GREEN)
    
    if scheduler:
        sched_type = config['training']['scheduler']['type']
        print_colored(f"Scheduler: {sched_type}", Colors.BRIGHT_GREEN)
    
    # Save config
    exp_id = config['experiment'].get('id', f"{model_name}_{loss_name}")
    save_dir = Path(config['checkpoint']['save_dir']) / exp_id
    save_config(config, save_dir / 'config.yaml')
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=config,
        scheduler=scheduler
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print_colored(f"\nResuming from checkpoint: {args.resume}", Colors.BRIGHT_YELLOW)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.history = checkpoint.get('history', trainer.history)
        trainer.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        trainer.best_epoch = checkpoint.get('best_epoch', 0)
        print_colored(f"Resumed from epoch {checkpoint['epoch']}", Colors.BRIGHT_GREEN)
    
    # Train
    history = trainer.train()
    
    # Plot training curves
    print_colored("\nGenerating training curves...", Colors.BRIGHT_CYAN)
    plot_path = save_dir / 'training_curves.png'
    plot_training_curves(
        history,
        save_path=plot_path,
        model_name=model_name,
        loss_name=loss_name
    )
    
    # Summary
    print_header("TRAINING COMPLETED")
    print_colored(f"✓ Best Val Acc: {trainer.best_val_acc:.2f}% (Epoch {trainer.best_epoch})", Colors.BRIGHT_GREEN, bold=True)
    print_colored(f"✓ Checkpoints: {save_dir}", Colors.BRIGHT_BLUE)
    print_colored(f"✓ Training curves: {plot_path}", Colors.BRIGHT_BLUE)
    print_colored(f"✓ TensorBoard logs: {save_dir / 'logs'}", Colors.BRIGHT_BLUE)
    print_colored(f"\nView logs: tensorboard --logdir {save_dir / 'logs'}\n", Colors.BRIGHT_YELLOW)


if __name__ == "__main__":
    main()