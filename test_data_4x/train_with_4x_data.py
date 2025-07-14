#!/usr/bin/env python3
"""
Train friend recommendation model with 4x larger dataset and optimized settings.
This script uses larger batch sizes and better hyperparameters for the bigger dataset.
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from datetime import datetime
import os
import sys
import argparse
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.append('../src')
from next_target_prediction_userids.next_target_prediction_userids import (
    NextTargetPredictionUserIDs, NextTargetPredictionBatch
)


class SocialNetworkDataset4X(Dataset):
    """
    Dataset for 4x larger social network data with optimized processing.
    """
    
    def __init__(self, actions: List[Dict], users: Dict, max_history_length: int = 32):
        self.actions = actions
        self.users = users
        self.max_history_length = max_history_length
        
        # Create action type mapping
        self.action_types = ['friend_request', 'friend_accept', 'message', 'like_post', 'comment']
        self.action_to_id = {action: i for i, action in enumerate(self.action_types)}
        
        # Build user histories
        print("Building user histories...")
        self.user_histories = self._build_user_histories()
        
        # Create training examples
        print("Creating training examples...")
        self.training_examples = self._create_training_examples()
        
        print(f"Created {len(self.training_examples)} training examples")
    
    def _build_user_histories(self) -> Dict[int, List[Tuple]]:
        """Build chronological histories for each user."""
        user_histories = {}
        
        for action in self.actions:
            actor_id = action['actor_id']
            if actor_id not in user_histories:
                user_histories[actor_id] = []
            
            user_histories[actor_id].append((
                self.action_to_id[action['action_type']],
                action['target_id'],
                action['timestamp']
            ))
        
        # Sort each user's history by timestamp
        for user_id in user_histories:
            user_histories[user_id].sort(key=lambda x: x[2])
        
        return user_histories
    
    def _create_training_examples(self) -> List[Dict]:
        """Create training examples from user histories."""
        examples = []
        
        for user_id, history in self.user_histories.items():
            if len(history) < 2:  # Need at least 2 actions for training
                continue
            
            # Create examples from each position in the history
            for i in range(1, len(history)):
                # Get the action we want to predict
                target_action, target_user, _ = history[i]
                
                # Get the history leading up to this action
                prev_history = history[:i]
                
                # Create training example
                example = {
                    'actor_id': user_id,
                    'history': prev_history,
                    'target_action': target_action,
                    'target_user': target_user
                }
                examples.append(example)
        
        return examples
    
    def __len__(self) -> int:
        return len(self.training_examples)
    
    def __getitem__(self, idx: int) -> NextTargetPredictionBatch:
        """Get a training example as a batch."""
        example = self.training_examples[idx]
        
        # Pad or truncate history
        history = example['history']
        if len(history) > self.max_history_length:
            history = history[-self.max_history_length:]  # Keep most recent
        
        # Create padded tensors
        history_length = len(history)
        padded_actions = [0] * self.max_history_length
        padded_targets = [0] * self.max_history_length
        padded_mask = [0] * self.max_history_length
        
        # Fill with actual data (most recent at the end)
        for i, (action, target, _) in enumerate(history):
            pos = self.max_history_length - history_length + i
            padded_actions[pos] = action
            padded_targets[pos] = target
            padded_mask[pos] = 1
        
        return NextTargetPredictionBatch(
            actor_id=torch.tensor([example['actor_id']], dtype=torch.long),
            actor_history_actions=torch.tensor([padded_actions], dtype=torch.long),
            actor_history_targets=torch.tensor([padded_targets], dtype=torch.long),
            actor_history_mask=torch.tensor([padded_mask], dtype=torch.float),
            example_action=torch.tensor([example['target_action']], dtype=torch.long),
            example_target=torch.tensor([example['target_user']], dtype=torch.long)
        )


def collate_fn(batch: List[NextTargetPredictionBatch]) -> NextTargetPredictionBatch:
    """Collate function for DataLoader."""
    return NextTargetPredictionBatch(
        actor_id=torch.cat([item.actor_id for item in batch]),
        actor_history_actions=torch.cat([item.actor_history_actions for item in batch]),
        actor_history_targets=torch.cat([item.actor_history_targets for item in batch]),
        actor_history_mask=torch.cat([item.actor_history_mask for item in batch]),
        example_action=torch.cat([item.example_action for item in batch]),
        example_target=torch.cat([item.example_target for item in batch])
    )


class Trainer4X:
    """
    Trainer optimized for 4x larger dataset with better monitoring.
    """
    
    def __init__(self, model: NextTargetPredictionUserIDs, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Optimized hyperparameters for larger dataset
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_mrrs = []
        self.val_mrrs = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_mrr = 0.0
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_mrr = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move batch to device
            batch = NextTargetPredictionBatch(
                actor_id=batch.actor_id.to(self.device),
                actor_history_actions=batch.actor_history_actions.to(self.device),
                actor_history_targets=batch.actor_history_targets.to(self.device),
                actor_history_mask=batch.actor_history_mask.to(self.device),
                example_action=batch.example_action.to(self.device),
                example_target=batch.example_target.to(self.device)
            )
            
            # Forward pass
            self.optimizer.zero_grad()
            results = self.model.train_forward(
                batch,
                num_rand_negs=10,  # More random negatives for larger dataset
                temporal_weight=0.3,  # Reduced temporal weight
                num_temporal_examples=6
            )
            
            # Backward pass
            loss = results['loss']
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_accuracy += results['accuracy'].item()
            total_mrr += results['mrr'].item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
            'mrr': total_mrr / num_batches
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_mrr = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = NextTargetPredictionBatch(
                    actor_id=batch.actor_id.to(self.device),
                    actor_history_actions=batch.actor_history_actions.to(self.device),
                    actor_history_targets=batch.actor_history_targets.to(self.device),
                    actor_history_mask=batch.actor_history_mask.to(self.device),
                    example_action=batch.example_action.to(self.device),
                    example_target=batch.example_target.to(self.device)
                )
                
                # Forward pass
                results = self.model.train_forward(
                    batch,
                    num_rand_negs=10,
                    temporal_weight=0.3,
                    num_temporal_examples=6
                )
                
                # Accumulate metrics
                total_loss += results['loss'].item()
                total_accuracy += results['accuracy'].item()
                total_mrr += results['mrr'].item()
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
            'mrr': total_mrr / num_batches
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 20) -> Dict[str, List[float]]:
        """Train the model."""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            start_time = datetime.now()
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Record metrics
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.train_accuracies.append(train_metrics['accuracy'])
            self.val_accuracies.append(val_metrics['accuracy'])
            self.train_mrrs.append(train_metrics['mrr'])
            self.val_mrrs.append(val_metrics['mrr'])
            
            # Save best model
            if val_metrics['mrr'] > self.best_val_mrr:
                self.best_val_mrr = val_metrics['mrr']
                torch.save(self.model.state_dict(), 'best_model_4x.pth')
                print(f"  Saved best model (MRR: {val_metrics['mrr']:.4f})")
            
            # Print progress
            epoch_time = (datetime.now() - start_time).total_seconds()
            print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s):")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Accuracy: {train_metrics['accuracy']:.4f}, "
                  f"MRR: {train_metrics['mrr']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Accuracy: {val_metrics['accuracy']:.4f}, "
                  f"MRR: {val_metrics['mrr']:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'train_mrrs': self.train_mrrs,
            'val_mrrs': self.val_mrrs
        }


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train friend recommendation model with 4X dataset')
    parser.add_argument('--interaction_type', type=str, default='mlp',
                       choices=['mlp', 'moe'],
                       help='Type of interaction modeling to use (default: mlp)')
    parser.add_argument('--num_experts', type=int, default=4,
                       help='Number of experts for MoE (default: 4)')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training (default: 128)')
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help='Embedding dimension (default: 256)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension (default: 512)')
    
    args = parser.parse_args()
    
    print("Friend Recommendation Model Training (4X Dataset)")
    print("=" * 60)
    print(f"Interaction Type: {args.interaction_type}")
    if args.interaction_type == 'moe':
        print(f"Number of Experts: {args.num_experts}")
    print(f"Number of Epochs: {args.num_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Embedding Dimension: {args.embedding_dim}")
    print(f"Hidden Dimension: {args.hidden_dim}")
    print("=" * 60)
    
    # Check for GPU
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using {device.upper()}")
    
    # Load data
    print("Loading social network data...")
    with open('social_network_data_4x.json', 'r') as f:
        data = json.load(f)
    
    users = data['users']
    actions = data['actions']
    
    print(f"Loaded {len(users):,} users and {len(actions):,} actions")
    
    # Create datasets
    print("Creating datasets...")
    dataset = SocialNetworkDataset4X(actions, users, max_history_length=32)
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"Dataset splits:")
    print(f"  Training: {len(train_dataset):,} examples")
    print(f"  Validation: {len(val_dataset):,} examples")
    print(f"  Test: {len(test_dataset):,} examples")
    
    # Create data loaders with larger batch sizes
    batch_size = args.batch_size
    num_workers = 4
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers
    )
    
    # Initialize model
    print("Initializing model...")
    num_users = len(users)
    num_actions = 5  # friend_request, friend_accept, message, like_post, comment
    
    model = NextTargetPredictionUserIDs(
        num_users=num_users,
        num_actions=num_actions,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_negatives=15,   # More negatives
        dropout=0.2,        # Slightly more dropout
        batch_size=batch_size,
        device=device,
        interaction_type=args.interaction_type,
        num_experts=args.num_experts
    )
    
    # Create trainer
    trainer = Trainer4X(model, device)
    
    # Train model
    print("Starting training...")
    training_history = trainer.train(train_loader, val_loader, num_epochs=args.num_epochs)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.validate(test_loader)
    
    print(f"\nTest Results:")
    print(f"- Loss: {test_metrics['loss']:.4f}")
    print(f"- Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"- MRR: {test_metrics['mrr']:.4f}")
    
    # Save training history
    with open('training_history_4x.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"- Best model saved to: best_model_4x.pth")
    print(f"- Training history saved to: training_history_4x.json")
    print(f"- Final test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"- Final test MRR: {test_metrics['mrr']:.4f}")


if __name__ == "__main__":
    main() 