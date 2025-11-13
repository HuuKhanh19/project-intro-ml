import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from datetime import datetime

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir='checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Tạo thư mục checkpoint nếu chưa có
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Lưu training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self):
        """Train một epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc='Training')
        
        for images, labels in pbar:
            # Chuyển data sang device (GPU/CPU)
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Tính metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'loss': running_loss / (pbar.n + 1),
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs, save_best=True):
        """Train model với số epochs nhất định"""
        print(f"Training on {self.device}")
        print(f"Total epochs: {num_epochs}")
        print("=" * 60)
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Lưu history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # In kết quả
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(
                    epoch=epoch,
                    filename='best_model.pth',
                    is_best=True
                )
                print(f"✅ Saved best model! (Val Acc: {val_acc:.2f}%)")
            
            # Save checkpoint mỗi 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(
                    epoch=epoch,
                    filename=f'checkpoint_epoch_{epoch+1}.pth'
                )
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        return self.history
    
    def save_checkpoint(self, epoch, filename='checkpoint.pth', is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']