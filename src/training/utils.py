"""
Training utility functions.
"""

import torch
import torch.optim as optim
from pathlib import Path


def create_optimizer(model, config):
    """
    Create optimizer based on configuration.
    
    Args:
        model: PyTorch model
        config: Configuration dictionary
        
    Returns:
        Optimizer instance
    """
    opt_config = config['training']['optimizer']
    opt_type = opt_config['type'].lower()
    lr = opt_config['lr']
    weight_decay = opt_config['weight_decay']
    
    if opt_type == 'adam':
        betas = tuple(opt_config.get('betas', [0.9, 0.999]))
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas
        )
    elif opt_type == 'adamw':
        betas = tuple(opt_config.get('betas', [0.9, 0.999]))
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas
        )
    elif opt_type == 'sgd':
        momentum = opt_config.get('momentum', 0.9)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")
    
    return optimizer


def create_scheduler(optimizer, config):
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer instance
        config: Configuration dictionary
        
    Returns:
        Scheduler instance or None
    """
    sched_config = config['training']['scheduler']
    sched_type = sched_config['type'].lower()
    
    if sched_type == 'none':
        return None
    
    params = sched_config.get('params', {})
    
    if sched_type == 'cosine':
        T_max = params.get('T_max', config['training']['epochs'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max
        )
    elif sched_type == 'step':
        step_size = params.get('step_size', 10)
        gamma = params.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif sched_type == 'plateau':
        patience = params.get('patience', 5)
        gamma = params.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=gamma,
            patience=patience,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")
    
    return scheduler


def save_checkpoint(state, save_dir, filename='checkpoint.pth'):
    """
    Save model checkpoint.
    
    Args:
        state: Dictionary containing model state and training info
        save_dir: Directory to save checkpoint
        filename: Checkpoint filename
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = save_dir / filename
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        
    Returns:
        Dictionary containing checkpoint info
    """
    checkpoint = torch.load(filepath)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


# Test functions
if __name__ == "__main__":
    from ..utils.config import load_config
    from ..models import create_model
    
    print("Testing training utilities...\n")
    
    # Load config
    config = load_config()
    
    # Create dummy model
    model = create_model('mlp', config)
    
    # Test optimizer creation
    print("1. Testing optimizer creation...")
    optimizer = create_optimizer(model, config)
    print(f"   Created optimizer: {type(optimizer).__name__}")
    print(f"   Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"   Weight decay: {optimizer.param_groups[0]['weight_decay']}")
    
    # Test scheduler creation
    print("\n2. Testing scheduler creation...")
    scheduler = create_scheduler(optimizer, config)
    if scheduler:
        print(f"   Created scheduler: {type(scheduler).__name__}")
    else:
        print("   No scheduler")
    
    # Test checkpoint saving/loading
    print("\n3. Testing checkpoint save/load...")
    
    state = {
        'epoch': 10,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': 85.5
    }
    
    save_checkpoint(state, 'test_checkpoint', 'test.pth')
    print("   ✓ Checkpoint saved")
    
    # Create new model and optimizer
    new_model = create_model('mlp', config)
    new_optimizer = create_optimizer(new_model, config)
    
    # Load checkpoint
    checkpoint = load_checkpoint('test_checkpoint/test.pth', new_model, new_optimizer)
    print(f"   ✓ Checkpoint loaded")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Best val acc: {checkpoint['best_val_acc']}")
    
    # Cleanup
    import shutil
    shutil.rmtree('test_checkpoint')
    print("   ✓ Cleanup done")
    
    print("\n✓ All training utility tests passed!")