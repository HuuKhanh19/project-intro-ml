"""
Training script with:
- Weighted Cross-Entropy Loss (ONLY for training, NOT for val/test)
- Early Stopping
- Learning Rate Scheduling
- Tensorboard Logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import time

from dataset import get_dataloaders
from models import get_model, count_parameters


class Trainer:
    def __init__(self, model_name, config_path='configs/config.yaml'):
        self.model_name = model_name
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create directories
        self.checkpoint_dir = Path(self.config['paths']['checkpoints']) / model_name
        self.log_dir = Path(self.config['paths']['logs']) / model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Get dataloaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(config_path)
        
        # Get model
        self.model = get_model(model_name, config_path).to(self.device)
        print(f"\nModel: {model_name}")
        print(f"Parameters: {count_parameters(self.model):,}")
        
        # Setup loss functions (CORRECTED - weighted for BOTH train & val)
        class_weights = torch.tensor(self.config['data']['class_weights']).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        print(f"\nClass weights (train & val): {class_weights.cpu().tolist()}")
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config['training']['lr_scheduler']['factor'],
            patience=self.config['training']['lr_scheduler']['patience'],
            min_lr=self.config['training']['lr_scheduler']['min_lr']
        )
        
        # Setup tensorboard
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.num_epochs = self.config['training']['num_epochs']
        self.patience = self.config['training']['patience']
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)  # Weighted
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        train_loss = total_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        
        return train_loss, train_acc
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]")
            
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)  # Weighted (same as training)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': total_loss / (pbar.n + 1),
                    'acc': 100. * correct / total
                })
        
        val_loss = total_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def save_checkpoint(self, epoch, val_loss, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
        }
        
        # Save last checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'last.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
            print(f"  → Saved best model (val_acc: {val_acc:.2f}%)")
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 80)
        print(f"TRAINING {self.model_name.upper()}")
        print("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning_rate', current_lr, epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # Check for improvement
            is_best = val_acc > self.best_val_acc
            
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, val_acc, is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                print(f"  Best val accuracy: {self.best_val_acc:.2f}%")
                break
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Training completed in {elapsed_time/3600:.2f} hours")
        print(f"  Best val accuracy: {self.best_val_acc:.2f}%")
        
        self.writer.close()
    
    def test(self):
        """Test model on test set with UNWEIGHTED metrics"""
        print("\n" + "=" * 80)
        print("TESTING (Unweighted Metrics for Clinical Evaluation)")
        print("=" * 80)
        
        # Load best checkpoint
        checkpoint = torch.load(self.checkpoint_dir / 'best.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
        
        self.model.eval()
        
        correct = 0
        total = 0
        class_correct = [0] * 5
        class_total = [0] * 5
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing")
            
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Per-class accuracy
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
        
        test_acc = 100. * correct / total
        print(f"\nOverall Test Accuracy: {test_acc:.2f}%")
        
        print("\nPer-class Accuracy:")
        class_names = self.config['data']['class_names']
        for i, class_name in enumerate(class_names):
            if class_total[i] > 0:
                acc = 100. * class_correct[i] / class_total[i]
                print(f"  {class_name}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        
        return test_acc


def main():
    parser = argparse.ArgumentParser(description='Train chest X-ray classification model')
    parser.add_argument('--model', type=str, required=True,
                      choices=['mlp', 'lenet', 'resnet18', 'efficientnet_b0'],
                      help='Model to train')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                      help='Path to config file')
    parser.add_argument('--test', action='store_true',
                      help='Run testing after training')
    
    args = parser.parse_args()
    
    # Train
    trainer = Trainer(args.model, args.config)
    trainer.train()
    
    # Test
    if args.test:
        trainer.test()


if __name__ == "__main__":
    main()