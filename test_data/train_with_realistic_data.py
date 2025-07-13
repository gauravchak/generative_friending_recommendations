#!/usr/bin/env python3
"""
Train the friend recommendation model with realistic social network data.

This script demonstrates how to train the model using the generated
social network data and evaluate its performance.
"""

import torch
import torch.optim as optim
import torch.nn as nn
import json
import os
import sys
import time
import argparse
from typing import Dict, List, Tuple
import numpy as np

# Add the parent directory to the path to import the model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'next_target_prediction_userids'))
from next_target_prediction_userids import NextTargetPredictionUserIDs
from data_loader import create_data_loaders, analyze_data


class Trainer:
    """Trainer for the friend recommendation model."""
    
    def __init__(
        self,
        model: NextTargetPredictionUserIDs,
        train_loader,
        val_loader,
        test_loader,
        learning_rate: float = 0.001,
        device: str = "cpu"
    ):
        """Initialize the trainer."""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            results = self.model.train_forward(
                batch, 
                num_rand_negs=3,  # Add some random negatives
                temporal_weight=0.3,  # Use temporal pretraining
                num_temporal_examples=8
            )
            
            loss = results['loss']
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_accuracy += results['accuracy'].item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_mrr = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                results = self.model.train_forward(
                    batch, 
                    num_rand_negs=3,
                    temporal_weight=0.3,
                    num_temporal_examples=8
                )
                
                # Accumulate metrics
                total_loss += results['loss'].item()
                total_accuracy += results['accuracy'].item()
                total_mrr += results['mrr'].item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_mrr = total_mrr / num_batches
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'mrr': avg_mrr
        }
    
    def _move_batch_to_device(self, batch):
        """Move batch tensors to the specified device."""
        batch.actor_id = batch.actor_id.to(self.device)
        batch.actor_history_actions = batch.actor_history_actions.to(self.device)
        batch.actor_history_targets = batch.actor_history_targets.to(self.device)
        batch.actor_history_mask = batch.actor_history_mask.to(self.device)
        batch.example_action = batch.example_action.to(self.device)
        batch.example_target = batch.example_target.to(self.device)
        return batch
    
    def train(
        self, 
        num_epochs: int = 10,
        early_stopping_patience: int = 5,
        save_path: str = None
    ) -> Dict[str, List[float]]:
        """Train the model."""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Record metrics
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.train_accuracies.append(train_metrics['accuracy'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s):")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}, MRR: {val_metrics['mrr']:.4f}")
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save best model
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': best_val_loss,
                        'val_accuracy': val_metrics['accuracy'],
                        'val_mrr': val_metrics['mrr']
                    }, save_path)
                    print(f"  Saved best model to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"  Early stopping after {epoch+1} epochs")
                    break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the test set."""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_mrr = 0.0
        total_mean_rank = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                results = self.model.train_forward(
                    batch, 
                    num_rand_negs=3,
                    temporal_weight=0.3,
                    num_temporal_examples=8
                )
                
                # Accumulate metrics
                total_loss += results['loss'].item()
                total_accuracy += results['accuracy'].item()
                total_mrr += results['mrr'].item()
                total_mean_rank += results['mean_rank'].item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_mrr = total_mrr / num_batches
        avg_mean_rank = total_mean_rank / num_batches
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'mrr': avg_mrr,
            'mean_rank': avg_mean_rank
        }


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train friend recommendation model')
    parser.add_argument('--history_encoder_type', type=str, default='transformer',
                       choices=['transformer', 'simple_attention'],
                       help='Type of history encoder to use (default: transformer)')
    parser.add_argument('--num_epochs', type=int, default=15,
                       help='Number of training epochs (default: 15)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Embedding dimension (default: 128)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension (default: 256)')
    
    args = parser.parse_args()
    
    print("Friend Recommendation Model Training")
    print("=" * 50)
    print(f"History Encoder Type: {args.history_encoder_type}")
    print(f"Number of Epochs: {args.num_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Embedding Dimension: {args.embedding_dim}")
    print(f"Hidden Dimension: {args.hidden_dim}")
    print("=" * 50)
    
    # Check if data exists
    data_file = "social_network_data.json"
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found!")
        print("Please run generate_social_network_data.py first.")
        return
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Analyze data
    print("\nAnalyzing training data...")
    analyze_data(data_file)
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_file=data_file,
        batch_size=args.batch_size,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1
    )
    
    # Get dataset statistics from the first batch
    for batch in train_loader:
        num_users = max(
            batch.actor_id.max().item(),
            batch.actor_history_targets.max().item(),
            batch.example_target.max().item()
        ) + 1
        num_actions = max(
            batch.actor_history_actions.max().item(),
            batch.example_action.max().item()
        ) + 1
        break
    
    print(f"\nDataset statistics:")
    print(f"- Number of users: {num_users}")
    print(f"- Number of actions: {num_actions}")
    
    # Initialize model
    print(f"\nInitializing model...")
    model = NextTargetPredictionUserIDs(
        num_users=num_users,
        num_actions=num_actions,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        device=device,
        history_encoder_type=args.history_encoder_type
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        device=device
    )
    
    # Train the model
    print(f"\nStarting training...")
    training_history = trainer.train(
        num_epochs=args.num_epochs,
        early_stopping_patience=5,
        save_path="test_data/best_model.pth"
    )
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    test_metrics = trainer.evaluate()
    
    print(f"\nTest Results:")
    print(f"- Loss: {test_metrics['loss']:.4f}")
    print(f"- Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"- MRR: {test_metrics['mrr']:.4f}")
    print(f"- Mean Rank: {test_metrics['mean_rank']:.2f}")
    
    # Save training history
    history_file = "test_data/training_history.json"
    with open(history_file, 'w') as f:
        json.dump({
            'train_losses': training_history['train_losses'],
            'val_losses': training_history['val_losses'],
            'train_accuracies': training_history['train_accuracies'],
            'val_accuracies': training_history['val_accuracies'],
            'test_metrics': test_metrics
        }, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"- Best model saved to: test_data/best_model.pth")
    print(f"- Training history saved to: {history_file}")
    print(f"- Final test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"- Final test MRR: {test_metrics['mrr']:.4f}")


if __name__ == "__main__":
    main() 