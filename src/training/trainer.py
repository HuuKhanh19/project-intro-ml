"""
Trainer class for model training and validation.
"""

import time
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..models import count_parameters
from .utils import save_checkpoint


class Trainer:
    """
    Training manager with logging, checkpointing, and early stopping.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        config: Configuration dictionary
        scheduler: Learning rate scheduler (optional)
    """
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 device, config, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # Get experiment info
        self.exp_name = config['experiment']['name']
        self.exp_id = config['experiment'].get('id', 'unknown')
        
        # Setup save directory
        save_dir = Path(config['checkpoint']['save_dir']) / self.exp_id
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        if config['logging']['use_tensorboard']:
            log_dir = self.save_dir / 'logs'
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
        
        # Training state
        self.current_epoch = 0
        self.total_epochs = config['training']['epochs']
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}/{self.total_epochs} [Train]')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate model."""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch}/{self.total_epochs} [Val]  ')
        
        with torch.no_grad():
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint_state(self, is_best=False):
        """Save model checkpoint."""
        state = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'history': self.history,
            'config': self.config
        }
        
        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        if self.config['checkpoint']['save_latest']:
            save_checkpoint(state, self.save_dir, 'checkpoint_latest.pth')
        
        # Save best checkpoint
        if is_best and self.config['checkpoint']['save_best']:
            save_checkpoint(state, self.save_dir, 'checkpoint_best.pth')
            print(f'  → Saved best model with val_acc: {self.best_val_acc:.2f}%')
        
        # Save periodic checkpoint
        save_freq = self.config['checkpoint'].get('save_frequency', 0)
        if save_freq > 0 and self.current_epoch % save_freq == 0:
            filename = f'checkpoint_epoch_{self.current_epoch}.pth'
            save_checkpoint(state, self.save_dir, filename)
    
    def log_metrics(self, train_loss, train_acc, val_loss, val_acc, lr):
        """Log metrics to TensorBoard."""
        if self.writer is not None:
            self.writer.add_scalar('Loss/train', train_loss, self.current_epoch)
            self.writer.add_scalar('Loss/val', val_loss, self.current_epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, self.current_epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, self.current_epoch)
            self.writer.add_scalar('Learning_rate', lr, self.current_epoch)
    
    def print_epoch_summary(self, train_loss, train_acc, val_loss, val_acc, lr):
        """Print epoch summary."""
        print(f"\nEpoch {self.current_epoch}/{self.total_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {lr:.6f}")
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*70)
        print(f"Training: {self.exp_name}")
        print("="*70)
        print(f"Experiment ID: {self.exp_id}")
        print(f"Device: {self.device}")
        print(f"Parameters: {count_parameters(self.model):,}")
        print(f"Epochs: {self.total_epochs}")
        print(f"Save directory: {self.save_dir}")
        print("="*70 + "\n")
        
        early_stopping_patience = self.config['training']['early_stopping_patience']
        start_time = time.time()
        
        for epoch in range(1, self.total_epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Log metrics
            self.log_metrics(train_loss, train_acc, val_loss, val_acc, current_lr)
            
            # Print summary
            self.print_epoch_summary(train_loss, train_acc, val_loss, val_acc, current_lr)
            
            # Check if best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
            
            # Save checkpoint
            self.save_checkpoint_state(is_best=is_best)
            
            # Early stopping
            if self.epochs_no_improve >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                print(f"  No improvement for {early_stopping_patience} consecutive epochs")
                break
            
            print("-" * 70)
        
        # Training completed
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("Training Completed!")
        print("="*70)
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        print(f"Checkpoints saved to: {self.save_dir}")
        print("="*70 + "\n")
        
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
        
        return self.history