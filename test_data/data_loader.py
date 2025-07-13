#!/usr/bin/env python3
"""
Data loader for social network training data.

This script loads the generated social network data and creates batches
for training the friend recommendation model.
"""

import torch
import json
import numpy as np
from typing import List, Dict, Iterator, Tuple
from torch.utils.data import Dataset, DataLoader
import random
from dataclasses import dataclass
import sys
import os
from datetime import datetime
from collections import defaultdict

# Add the parent directory to the path to import the model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'next_target_prediction_userids'))
from next_target_prediction_userids import NextTargetPredictionBatch


def process_raw_actions_to_examples(data_file: str, max_history_length: int = 50) -> List[Dict]:
    """
    Process raw actions into training examples for next-target prediction.
    
    Args:
        data_file: Path to the social network data JSON file
        max_history_length: Maximum length of user interaction history
        
    Returns:
        List of training examples
    """
    print(f"Processing raw actions from {data_file}...")
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    actions = data['actions']
    users = data['users']
    
    # Group actions by actor
    user_actions = defaultdict(list)
    for action in actions:
        actor_id = action['actor_id']
        user_actions[actor_id].append(action)
    
    # Sort actions by timestamp for each user
    for actor_id in user_actions:
        user_actions[actor_id].sort(key=lambda x: x['timestamp'])
    
    # Create training examples
    examples = []
    
    # Action type mapping
    action_type_map = {"follow": 0, "like": 1, "comment": 2, "share": 3}
    
    for actor_id, actions_list in user_actions.items():
        if len(actions_list) < 2:
            continue  # Skip users with insufficient history
        
        # Create examples from user's action history
        for i in range(len(actions_list) - 1):
            # Get history up to current action
            history_actions = actions_list[:i + 1]
            
            # Get next action as target
            next_action = actions_list[i + 1]
            
            # Convert action types to integers
            history_action_types = [action_type_map[action['action_type']] for action in history_actions]
            history_targets = [action['target_id'] for action in history_actions]
            
            # Pad history if necessary
            while len(history_action_types) < max_history_length:
                history_action_types.insert(0, 0)  # padding
                history_targets.insert(0, 0)  # padding
            
            # Truncate if too long
            if len(history_action_types) > max_history_length:
                history_action_types = history_action_types[-max_history_length:]
                history_targets = history_targets[-max_history_length:]
            
            # Create history mask
            history_mask = [0] * max_history_length
            valid_length = min(len(actions_list[:i + 1]), max_history_length)
            for j in range(valid_length):
                history_mask[-(j+1)] = 1  # Valid entries at the end
            
            # Create example
            example = {
                'actor_id': actor_id,
                'actor_history_actions': history_action_types,
                'actor_history_targets': history_targets,
                'actor_history_mask': history_mask,
                'example_action': action_type_map[next_action['action_type']],
                'example_target': next_action['target_id']
            }
            
            examples.append(example)
    
    print(f"Created {len(examples)} training examples from {len(actions)} actions")
    return examples


class SocialNetworkDataset(Dataset):
    """Dataset for social network training data."""
    
    def __init__(self, data_file: str, max_history_length: int = 50):
        """
        Initialize the dataset.
        
        Args:
            data_file: Path to the social network data JSON file
            max_history_length: Maximum length of user interaction history
        """
        self.max_history_length = max_history_length
        
        # Process raw actions into training examples
        self.examples = process_raw_actions_to_examples(data_file, max_history_length)
        
        print(f"Loaded {len(self.examples)} training examples")
        
        # Extract unique user and action IDs for statistics
        all_user_ids = set()
        all_action_types = set()
        
        for example in self.examples:
            all_user_ids.add(example['actor_id'])
            all_user_ids.add(example['example_target'])
            all_action_types.add(example['example_action'])
            
            # Add history targets
            for target_id in example['actor_history_targets']:
                if target_id > 0:  # Skip padding
                    all_user_ids.add(target_id)
            
            # Add history actions
            for action_type in example['actor_history_actions']:
                if action_type > 0:  # Skip padding
                    all_action_types.add(action_type)
        
        self.num_users = max(all_user_ids) + 1
        self.num_actions = max(all_action_types) + 1
        
        print(f"Dataset statistics:")
        print(f"- Number of unique users: {self.num_users}")
        print(f"- Number of action types: {self.num_actions}")
        print(f"- Number of training examples: {len(self.examples)}")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> NextTargetPredictionBatch:
        """Get a single training example."""
        example = self.examples[idx]
        
        # Convert to tensors
        actor_id = torch.tensor(example['actor_id'], dtype=torch.long)
        actor_history_actions = torch.tensor(example['actor_history_actions'], dtype=torch.long)
        actor_history_targets = torch.tensor(example['actor_history_targets'], dtype=torch.long)
        actor_history_mask = torch.tensor(example['actor_history_mask'], dtype=torch.long)
        example_action = torch.tensor(example['example_action'], dtype=torch.long)
        example_target = torch.tensor(example['example_target'], dtype=torch.long)
        
        return NextTargetPredictionBatch(
            actor_id=actor_id,
            actor_history_actions=actor_history_actions,
            actor_history_targets=actor_history_targets,
            actor_history_mask=actor_history_mask,
            example_action=example_action,
            example_target=example_target
        )


def collate_fn(batch: List[NextTargetPredictionBatch]) -> NextTargetPredictionBatch:
    """Collate function for batching."""
    # Stack all tensors
    actor_ids = torch.stack([item.actor_id for item in batch])
    actor_history_actions = torch.stack([item.actor_history_actions for item in batch])
    actor_history_targets = torch.stack([item.actor_history_targets for item in batch])
    actor_history_masks = torch.stack([item.actor_history_mask for item in batch])
    example_actions = torch.stack([item.example_action for item in batch])
    example_targets = torch.stack([item.example_target for item in batch])
    
    return NextTargetPredictionBatch(
        actor_id=actor_ids,
        actor_history_actions=actor_history_actions,
        actor_history_targets=actor_history_targets,
        actor_history_mask=actor_history_masks,
        example_action=example_actions,
        example_target=example_targets
    )


def create_data_loaders(
    data_file: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    max_history_length: int = 50,
    shuffle: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/validation/test data loaders.
    
    Args:
        data_file: Path to the social network data JSON file
        batch_size: Batch size for training
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        max_history_length: Maximum length of user interaction history
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load dataset
    dataset = SocialNetworkDataset(data_file, max_history_length)
    
    # Calculate split indices
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Create splits
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    print(f"Data splits:")
    print(f"- Training: {len(train_dataset)} examples ({len(train_loader)} batches)")
    print(f"- Validation: {len(val_dataset)} examples ({len(val_loader)} batches)")
    print(f"- Test: {len(test_dataset)} examples ({len(test_loader)} batches)")
    
    return train_loader, val_loader, test_loader


def analyze_data(data_file: str):
    """Analyze the generated data."""
    print("Analyzing social network data...")
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    actions = data['actions']
    users = data['users']
    
    # Basic statistics
    print(f"Total users: {len(users)}")
    print(f"Total actions: {len(actions)}")
    
    # Action type distribution
    action_counts = defaultdict(int)
    for action in actions:
        action_counts[action['action_type']] += 1
    
    print(f"\nAction type distribution:")
    for action_type, count in action_counts.items():
        percentage = (count / len(actions)) * 100
        print(f"- {action_type}: {count} ({percentage:.1f}%)")
    
    # User activity distribution
    user_action_counts = defaultdict(int)
    for action in actions:
        user_action_counts[action['actor_id']] += 1
    
    activity_counts = list(user_action_counts.values())
    print(f"\nUser activity statistics:")
    print(f"- Average actions per user: {np.mean(activity_counts):.2f}")
    print(f"- Median actions per user: {np.median(activity_counts):.2f}")
    print(f"- Min actions per user: {min(activity_counts)}")
    print(f"- Max actions per user: {max(activity_counts)}")
    
    # Popularity distribution
    popularities = [user['final_popularity'] for user in users.values()]
    print(f"\nPopularity statistics:")
    print(f"- Average popularity: {np.mean(popularities):.2f}")
    print(f"- Median popularity: {np.median(popularities):.2f}")
    print(f"- Min popularity: {min(popularities)}")
    print(f"- Max popularity: {max(popularities)}")


def main():
    """Test the data loader."""
    print("Social Network Data Loader Test")
    print("=" * 40)
    
    data_file = "social_network_data.json"
    
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found!")
        print("Please run generate_social_network_data.py first.")
        return
    
    # Analyze the data
    analyze_data(data_file)
    
    print("\n" + "=" * 40)
    print("Testing data loader...")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_file=data_file,
        batch_size=16,
        max_history_length=20
    )
    
    # Test a batch
    print("\nTesting batch processing...")
    for batch in train_loader:
        print(f"Batch shapes:")
        print(f"- actor_id: {batch.actor_id.shape}")
        print(f"- actor_history_actions: {batch.actor_history_actions.shape}")
        print(f"- actor_history_targets: {batch.actor_history_targets.shape}")
        print(f"- actor_history_mask: {batch.actor_history_mask.shape}")
        print(f"- example_action: {batch.example_action.shape}")
        print(f"- example_target: {batch.example_target.shape}")
        break
    
    print("\n✅ Data loader test completed successfully!")


if __name__ == "__main__":
    main() 