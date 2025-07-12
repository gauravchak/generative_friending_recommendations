"""
Test file for Next Target Prediction using UserIDs model.

This file demonstrates how to use the NextTargetPredictionUserIDs model
with sample data and shows the expected behavior.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List
import sys
import os

# Add project root to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our model
from src.next_target_prediction_userids.next_target_prediction_userids import NextTargetPredictionUserIDs, NextTargetPredictionBatch


def create_sample_batch(
    num_users: int = 1000,
    num_actions: int = 10,
    history_length: int = 128,
    model=None,
) -> NextTargetPredictionBatch:
    """
    Create a sample batch for testing.
    
    Args:
        num_users: Total number of users in the system
        num_actions: Total number of actions
        history_length: Length of user history
        model: The model instance to get batch_size and device
        
    Returns:
        NextTargetPredictionBatch: Sample batch for testing
    """
    batch_size = model.batch_size if model is not None else 4
    device = model.device if model is not None else "cpu"
    # Create sample actor IDs
    actor_id = torch.randint(0, num_users, (batch_size,), device=device)
    
    # Create sample history - some users have shorter histories
    actor_history_actions = torch.randint(0, num_actions, (batch_size, history_length), device=device)
    actor_history_targets = torch.randint(0, num_users, (batch_size, history_length), device=device)
    
    # Create masks - simulate variable length histories
    actor_history_mask = torch.ones(batch_size, history_length, device=device)
    for i in range(batch_size):
        # Random history length between 10 and history_length
        valid_length = torch.randint(10, history_length, (1,)).item()
        actor_history_mask[i, valid_length:] = 0
    
    # Create example actions and targets
    example_action = torch.randint(0, num_actions, (batch_size,), device=device)
    example_target = torch.randint(0, num_users, (batch_size,), device=device)
    
    return NextTargetPredictionBatch(
        actor_id=actor_id,
        actor_history_actions=actor_history_actions,
        actor_history_targets=actor_history_targets,
        actor_history_mask=actor_history_mask,
        example_action=example_action,
        example_target=example_target,
    )


def test_model_initialization():
    """Test that the model can be initialized properly."""
    print("Testing model initialization...")
    
    model = NextTargetPredictionUserIDs(
        num_users=1000,
        num_actions=10,
        embedding_dim=64,
        hidden_dim=128,
        num_negatives=5,
        dropout=0.1,
        batch_size=8,
        device="cpu",
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print("✓ Model initialization successful")


def test_forward_pass():
    """Test that the model can perform forward pass."""
    print("\nTesting forward pass...")
    
    model = NextTargetPredictionUserIDs(
        num_users=1000,
        num_actions=10,
        embedding_dim=64,
        hidden_dim=128,
        num_negatives=5,
        dropout=0.1,
        batch_size=8,
        device="cpu",
    )
    
    batch = create_sample_batch(num_users=1000, num_actions=10, model=model)
    
    # Test forward pass - now returns actor-action representation
    with torch.no_grad():
        actor_action_repr = model.forward(
            batch.actor_id,
            batch.actor_history_actions,
            batch.actor_history_targets,
            batch.actor_history_mask,
            batch.example_action,
        )
    
    print(f"Forward pass output shape: {actor_action_repr.shape}")
    print(f"Sample actor-action representations: {actor_action_repr[:3]}")
    print("✓ Forward pass successful")


def test_train_forward():
    """Test the training forward pass with different negative sampling strategies."""
    print("\nTesting train_forward (mixed negative sampling)...")
    
    model = NextTargetPredictionUserIDs(
        num_users=1000,
        num_actions=10,
        embedding_dim=64,
        hidden_dim=128,
        num_negatives=5,
        dropout=0.1,
        batch_size=8,
        device="cpu",
    )
    
    batch = create_sample_batch(num_users=1000, num_actions=10, model=model)
    
    # Test pure in-batch negative sampling (K=0)
    print("Testing pure in-batch negative sampling (K=0):")
    results = model.train_forward(batch, K=0)
    
    print("Training metrics (Pure In-Batch Negative Sampling):")
    for key, value in results.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Check that loss is a scalar
    assert results['loss'].dim() == 0, "Loss should be a scalar"
    assert results['loss'].item() >= 0, "Loss should be non-negative"
    assert results['num_negatives'].item() == 7, "Should have 7 in-batch negatives (batch_size - 1)"
    
    # Test mixed negative sampling (K=3)
    print("\nTesting mixed negative sampling (K=3):")
    results_mixed = model.train_forward(batch, K=3)
    
    print("Training metrics (Mixed Negative Sampling):")
    for key, value in results_mixed.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Check that loss is a scalar
    assert results_mixed['loss'].dim() == 0, "Loss should be a scalar"
    assert results_mixed['loss'].item() >= 0, "Loss should be non-negative"
    assert results_mixed['num_negatives'].item() == 10, "Should have 10 total negatives (7 in-batch + 3 random)"
    
    print("✓ train_forward successful")


def test_training_loop():
    """Test a simple training loop to ensure gradients flow properly."""
    print("\nTesting training loop...")
    
    model = NextTargetPredictionUserIDs(
        num_users=1000,
        num_actions=10,
        embedding_dim=64,
        hidden_dim=128,
        num_negatives=5,
        dropout=0.1,
        batch_size=8,
        device="cpu",
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop with mixed negative sampling
    initial_loss = None
    for step in range(5):
        batch = create_sample_batch(num_users=1000, num_actions=10, model=model)
        
        optimizer.zero_grad()
        results = model.train_forward(batch, K=2)  # Use 2 additional random negatives
        loss = results['loss']
        
        if initial_loss is None:
            initial_loss = loss.item()
        
        loss.backward()
        optimizer.step()
        
        print(f"Step {step + 1}: Loss = {loss.item():.4f}, Accuracy = {results['accuracy'].item():.4f}, Negatives = {results['num_negatives'].item()}")
    
    print(f"Initial loss: {initial_loss:.4f}")
    print("✓ Training loop successful")


def test_prediction():
    """Test the prediction functionality."""
    print("\nTesting prediction...")
    
    model = NextTargetPredictionUserIDs(
        num_users=1000,
        num_actions=10,
        embedding_dim=64,
        hidden_dim=128,
        num_negatives=5,
        dropout=0.1,
        batch_size=8,
        device="cpu",
    )
    
    # Create sample input for prediction
    batch_size = model.batch_size
    num_candidates = 100
    
    actor_id = torch.randint(0, 1000, (batch_size,), device=model.device)
    actor_history_actions = torch.randint(0, 10, (batch_size, 128), device=model.device)
    actor_history_targets = torch.randint(0, 1000, (batch_size, 128), device=model.device)
    actor_history_mask = torch.ones(batch_size, 128, device=model.device)
    action_id = torch.randint(0, 10, (batch_size,), device=model.device)
    candidate_targets = torch.randint(0, 1000, (batch_size, num_candidates), device=model.device)
    
    # Test prediction
    with torch.no_grad():
        top_k_scores, top_k_indices = model.predict_top_k(
            actor_id=actor_id,
            actor_history_actions=actor_history_actions,
            actor_history_targets=actor_history_targets,
            actor_history_mask=actor_history_mask,
            action_id=action_id,
            candidate_targets=candidate_targets,
            k=10,
        )
    
    print(f"Top-k scores shape: {top_k_scores.shape}")
    print(f"Top-k indices shape: {top_k_indices.shape}")
    print(f"Sample top-3 scores for first example: {top_k_scores[0, :3]}")
    print("✓ Prediction successful")


def run_all_tests():
    """Run all tests."""
    print("Running Next Target Prediction UserIDs Tests")
    print("=" * 50)
    
    test_model_initialization()
    test_forward_pass()
    test_train_forward()
    test_training_loop()
    test_prediction()
    
    print("\n" + "=" * 50)
    print("All tests completed successfully! ✓")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    run_all_tests() 