#!/usr/bin/env python3
"""
Unified training script for friend recommendation model.

This script can train on both the regular dataset (test_data/regular) and the 4x larger dataset (test_data/4x)
by specifying the --dataset argument.
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
from datetime import datetime
from torch.utils.data import DataLoader, Dataset

# Add the src directory to the path to import the model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'next_target_prediction_userids'))
from next_target_prediction_userids import NextTargetPredictionUserIDs, NextTargetPredictionBatch

# For regular dataset
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'regular'))
    from data_loader import create_data_loaders, analyze_data
except ImportError:
    # If data_loader is not available, we'll use the 4x dataset approach
    pass


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


class Trainer:
    """Unified trainer for the friend recommendation model."""
    
    def __init__(
        self,
        model: NextTargetPredictionUserIDs,
        train_loader,
        val_loader,
        test_loader,
        learning_rate: float = 0.001,
        device: str = "cpu",
        use_adamw: bool = False
    ):
        """Initialize the trainer."""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Optimizer
        if use_adamw:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=0.01,
                betas=(0.9, 0.999)
            )
        else:
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
        self.train_mrrs = []
        self.val_mrrs = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_mrr = 0.0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_mrr = 0.0
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
            self.train_mrrs.append(train_metrics['mrr'])
            self.val_mrrs.append(val_metrics['mrr'])
            
            # Save best model based on MRR
            if val_metrics['mrr'] > self.best_val_mrr:
                self.best_val_mrr = val_metrics['mrr']
                patience_counter = 0
                
                # Save best model
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_metrics['loss'],
                        'val_accuracy': val_metrics['accuracy'],
                        'val_mrr': val_metrics['mrr']
                    }, save_path)
                    print(f"  Saved best model to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"  Early stopping after {epoch+1} epochs")
                    break
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s):")
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


def load_4x_data(data_dir: str):
    """Load 4x dataset data."""
    data_file = os.path.join(data_dir, 'social_network_data_4x.json')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file {data_file} not found!")
    
    print(f"Loading data from {data_file}...")
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    users = data['users']
    actions = data['actions']
    
    print(f"Loaded {len(users):,} users and {len(actions):,} actions")
    return users, actions


def create_4x_data_loaders(data_dir: str, batch_size: int, num_workers: int = 4):
    """Create data loaders for 4x dataset."""
    users, actions = load_4x_data(data_dir)
    
    # Create dataset
    print("Creating dataset...")
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
    
    # Create data loaders
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
    
    return train_loader, val_loader, test_loader, len(users), 5


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train friend recommendation model')
    parser.add_argument('--dataset', type=str, default='regular',
                       choices=['regular', '4x'],
                       help='Dataset to use (default: regular)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory containing the dataset (default: auto-detect)')
    parser.add_argument('--history_encoder_type', type=str, default='transformer',
                       choices=['transformer', 'simple_attention'],
                       help='Type of history encoder to use (default: transformer)')
    parser.add_argument('--interaction_type', type=str, default='mlp',
                       choices=['mlp', 'moe'],
                       help='Type of interaction modeling to use (default: mlp)')
    parser.add_argument('--num_experts', type=int, default=4,
                       help='Number of experts for MoE (default: 4)')
    parser.add_argument('--num_epochs', type=int, default=15,
                       help='Number of training epochs (default: 15)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training (default: auto-detect)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--embedding_dim', type=int, default=None,
                       help='Embedding dimension (default: auto-detect)')
    parser.add_argument('--hidden_dim', type=int, default=None,
                       help='Hidden dimension (default: auto-detect)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for models and logs (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Auto-detect data directory
    if args.data_dir is None:
        if args.dataset == '4x':
            args.data_dir = os.path.join(os.path.dirname(__file__), '4x')
        else:
            args.data_dir = os.path.join(os.path.dirname(__file__), 'regular')
    
    # Auto-detect output directory
    if args.output_dir is None:
        args.output_dir = args.data_dir
    
    # Auto-detect hyperparameters based on dataset
    if args.batch_size is None:
        args.batch_size = 128 if args.dataset == '4x' else 32
    
    if args.embedding_dim is None:
        args.embedding_dim = 256 if args.dataset == '4x' else 128
    
    if args.hidden_dim is None:
        args.hidden_dim = 512 if args.dataset == '4x' else 256
    
    print("Friend Recommendation Model Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"History Encoder Type: {args.history_encoder_type}")
    print(f"Interaction Type: {args.interaction_type}")
    if args.interaction_type == 'moe':
        print(f"Number of Experts: {args.num_experts}")
    print(f"Number of Epochs: {args.num_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Embedding Dimension: {args.embedding_dim}")
    print(f"Hidden Dimension: {args.hidden_dim}")
    print("=" * 60)
    
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
    
    # Create data loaders
    print("\nCreating data loaders...")
    if args.dataset == '4x':
        train_loader, val_loader, test_loader, num_users, num_actions = create_4x_data_loaders(
            args.data_dir, args.batch_size
        )
    else:
        # Use regular dataset
        data_file = os.path.join(args.data_dir, "social_network_data.json")
        if not os.path.exists(data_file):
            print(f"Data file {data_file} not found!")
            print("Please run generate_social_network_data.py first.")
            return
        
        # Analyze data
        print("\nAnalyzing training data...")
        analyze_data(data_file)
        
        # Create data loaders
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
        history_encoder_type=args.history_encoder_type,
        interaction_type=args.interaction_type,
        num_experts=args.num_experts
    )
    
    # Initialize trainer
    use_adamw = args.dataset == '4x'  # Use AdamW for 4x dataset
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        device=device,
        use_adamw=use_adamw
    )
    
    # Train the model
    print(f"\nStarting training...")
    model_suffix = "_4x" if args.dataset == '4x' else ""
    save_path = os.path.join(args.output_dir, f"best_model{model_suffix}.pth")
    
    training_history = trainer.train(
        num_epochs=args.num_epochs,
        early_stopping_patience=5,
        save_path=save_path
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
    history_file = os.path.join(args.output_dir, f"training_history{model_suffix}.json")
    with open(history_file, 'w') as f:
        json.dump({
            'train_losses': training_history['train_losses'],
            'val_losses': training_history['val_losses'],
            'train_accuracies': training_history['train_accuracies'],
            'val_accuracies': training_history['val_accuracies'],
            'train_mrrs': training_history['train_mrrs'],
            'val_mrrs': training_history['val_mrrs'],
            'test_metrics': test_metrics
        }, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"- Best model saved to: {save_path}")
    print(f"- Training history saved to: {history_file}")
    print(f"- Final test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"- Final test MRR: {test_metrics['mrr']:.4f}")


if __name__ == "__main__":
    main() 